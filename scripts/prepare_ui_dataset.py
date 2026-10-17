#!/usr/bin/env python3
"""Build local UI fixtures from training annotations. Never runs an AI model.
Requires ffmpeg/ffprobe and PyYAML. Media and results go in ignored data/.
--seed-local imports only into the named local demo Docker stack, without replacing jobs.
"""
import argparse
import base64
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import uuid
import yaml

CASES = ['Х-4', 'Х-26', 'Х-32', 'Х-40', 'Х-46', 'Х-3', 'Х-12', 'Х-35']

def run(*args):
    return subprocess.check_output(args, stderr=subprocess.PIPE)

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('data/ui-fixtures'))
    ap.add_argument('--seed-local', action='store_true')
    args = ap.parse_args()
    source = args.source.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    annotations = json.loads((source/'configs/train_annotations.json').read_text())
    split = json.loads((source/'configs/split.json').read_text())
    rules = yaml.safe_load((source/'app/rules/norms.yaml').read_text())
    labels = {r['наименование файла']: r for r in csv.DictReader((source/'data/labels_all.csv').open())}
    videos = {p.stem: p for p in (source/'data/train').rglob('*') if p.suffix.lower() in {'.mov', '.mp4'}}
    jobs = []; manifest = []
    anchor = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
    for i, name in enumerate(CASES):
        assert name in split['train'] and name not in split['duplicates'], name
        src = videos[name]; annotation = annotations[name]
        expected = 'no_violation' if labels[name]['нарушение / без нарушения'] == 'без нарушения' else 'violation'
        assert annotation['label'] == expected
        duration = float(run('ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(src)))
        job_id = uuid.uuid5(uuid.NAMESPACE_URL, 'cashguard:ui-dataset:v1:'+name).hex
        video = out/(job_id+'.mp4')
        if not video.exists():
            # Browser-friendly local copy, original duration/timeline preserved. No external services.
            run('ffmpeg','-y','-v','error','-i',str(src),'-map','0:v:0','-vf',"scale=960:960:force_original_aspect_ratio=decrease:force_divisible_by=2",'-c:v','libx264','-preset','fast','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(video))
        events=[]
        for event in annotation['events']:
            t=float(event['t']); assert 0 <= t < duration, (name,t,duration)
            rule=rules[event['type_id']]
            frames=[]
            for stamp in [t, min(t+2, max(t,duration-.1))]:
                data=run('ffmpeg','-v','error','-ss',str(stamp),'-i',str(src),'-frames:v','1','-vf','scale=640:640:force_original_aspect_ratio=decrease','-f','image2pipe','-c:v','mjpeg','-q:v','4','-')
                assert data[:2] == b'\xff\xd8'
                frames.append({'timestamp_sec':stamp,'image_base64':base64.b64encode(data).decode()})
            events.append({'type':rule['title'],'timecode_sec':t,'description':'ТЕСТ ИНТЕРФЕЙСА · По ручной разметке команды: '+annotation['note']+'. Это ожидаемый результат для проверки UI, не вывод модели. Кадры извлечены из исходной записи. Нормативная ссылка перенесена из справочника проекта без дополнительной проверки.' + (' Связь с нормой косвенная.' if rule.get('interpretive') else ''),'evidence_frames':frames,'rule':{'title':rule['document'],'section':rule['clause'],'url':rule['url']}})
        job={'job_id':job_id,'status':'done','verdict':expected,'violations':events,'filename':f'[DATASET TEST] {name}.mp4','created_at':(anchor-timedelta(days=i%7,hours=i)).isoformat(),'demo':True}
        jobs.append(job)
        manifest.append({'job_id':job_id,'source_video':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'duration_sec':duration,'organizer_label':expected,'organizer_timecode':labels[name]['таймстемп'],'event_origin':'team manual annotations; extra events are not organizer labels','type_ids':[e['type_id'] for e in annotation['events']],'note':annotation['note'],'video':video.name,'created_at_is_synthetic':True})
        dump(out/(job_id+'.json'),job)
        print(name, expected, len(events), 'events', flush=True)
    dump(out/'jobs.json',jobs)
    dump(out/'manifest.json',{'purpose':'UI testing only, not model evaluation','created_at_note':'Dates distributed synthetically over one week for dashboard testing.','cases':manifest})
    # API states for isolated tests; not seeded as permanently running jobs.
    states=[]
    for state in ['queued','processing','failed']:
        j={**jobs[0], 'status':state,'violations':[]};j.pop('verdict',None)
        if state=='failed': j['error']='Искусственная ошибка для проверки интерфейса. Исходное видео исправно.'
        states.append(j)
    dump(out/'state-fixtures.json',states)
    (out/'README.txt').write_text('LOCAL UI TEST DATA. Not AI predictions.\nOnly train videos; no held-out test clips.\nLabels from data/labels_all.csv; event types, additional events and times from configs/train_annotations.json.\nEvidence JPEGs extracted locally from source video. Normative references copied from app/rules/norms.yaml, not revalidated.\ncreated_at dates are synthetic to populate the chart.\nstate-fixtures.json: isolated queued/processing/failed snapshots; not seeded.\nDo not publish dataset media or frames. This directory is gitignored.\n')
    if args.seed_local:
        health=json.load(urllib.request.urlopen('http://127.0.0.1:8081/api/health'))
        assert health.get('demo') is True, 'Only local demo server may be seeded'
        container='cashguard-ui-backend-1';db='cashguard-ui-db-1'
        sql=['BEGIN;']
        def q(value): return "'"+value.replace("'","''")+"'"
        for job in jobs:
            vid=out/(job['job_id']+'.mp4');dest='/data/uploads/'+vid.name
            subprocess.run(['docker','cp',str(vid),container+':'+dest],check=True,stdout=subprocess.DEVNULL)
            row=[job['job_id'],'jury',dest,json.dumps(job,ensure_ascii=False),job['created_at']]
            sql.append('INSERT INTO jobs(id,owner_id,path,result,created_at) VALUES('+','.join(map(q,row))+') ON CONFLICT(id) DO NOTHING;')
        sql.append('COMMIT;')
        subprocess.run(['docker','exec','-i',db,'psql','-U','cashguard','-d','cashguard','-v','ON_ERROR_STOP=1'],input='\n'.join(sql).encode(),check=True,stdout=subprocess.DEVNULL)
        print('Seeded 8 fixture jobs into local demo stack. Existing jobs retained.')
    print('Output:',out)

if __name__=='__main__': main()
