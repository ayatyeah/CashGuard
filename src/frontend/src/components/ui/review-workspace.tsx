import { useRef, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { ArrowDownToLine, ArrowLeft, Check, Clock3, FileVideo, Maximize2, Play, Plus, ShieldAlert, ShieldCheck, X } from 'lucide-react';
import { Button } from './button';
import { StoredVideo } from './stored-video';
import { timecode, type Job } from '@/lib/api';
import { useWorkspace } from '@/lib/workspace';

export function ReviewWorkspace({ job, localURL, onClose, onNew, onReport }: { job: Job; localURL?: string; onClose: () => void; onNew: () => void; onReport: () => void }) {
  const { t } = useWorkspace();
  const video = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<number | null>(null);
  const [selected, setSelected] = useState(0);
  const [duration, setDuration] = useState(0);
  const [position, setPosition] = useState(0);
  const [seekError, setSeekError] = useState(false);
  const [expanded, setExpanded] = useState<{ src: string; label: string } | null>(null);
  const active = job.status === 'queued' || job.status === 'processing';
  const event = job.violations[selected];
  const seek = (seconds: number) => {
    const player = video.current;
    if (!player || !Number.isFinite(player.duration)) { pendingSeek.current = seconds; return; }
    if (seconds < 0 || seconds > player.duration) { setSeekError(true); return; }
    setSeekError(false); player.currentTime = seconds; setPosition(seconds); pendingSeek.current = null;
  };
  const choose = (index: number) => { setSelected(index); seek(job.violations[index].timecode_sec); };
  const onReady = () => {
    const length = video.current?.duration ?? 0;
    setDuration(Number.isFinite(length) ? length : 0);
    if (pendingSeek.current !== null) seek(pendingSeek.current);
  };
  return <section className="review-workspace" aria-labelledby="review-title">
    <div className="review-heading">
      <div className="min-w-0"><button className="review-back" onClick={onClose}><ArrowLeft size={14}/>{t('История анализов')}</button>
        <h2 id="review-title">{job.filename}</h2>
        <p>{t('От видео к решению')}{job.demo && <span className="review-demo">{t('ДЕМО')}</span>}</p>
      </div>
      <div className="flex flex-wrap gap-2">
        {job.status === 'done' && <Button size="sm" variant="outline" onClick={onReport}><ArrowDownToLine size={15}/>{t('Скачать JSON')}</Button>}
        <Button size="sm" variant="outline" onClick={onNew}><Plus size={15}/>{t('Новый анализ')}</Button>
      </div>
    </div>
    {active && <ProcessingState status={job.status} />}
    {job.status === 'failed' && <div className="review-verdict failed" role="alert"><ShieldAlert/><div><strong>{t('Ошибка обработки')}</strong><p>{t(job.error || 'Попробуйте загрузить видео ещё раз.')}</p></div></div>}
    {job.status === 'done' && !!job.warnings?.length && <div className="review-verdict flagged" role="status"><ShieldAlert/><div><strong>{t('Ограничения анализа')}</strong><ul>{job.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul><p>{t('Учитывайте ограничения при проверке результата.')}</p></div></div>}
    {job.status === 'done' && <div className={`review-verdict ${job.verdict === 'violation' || job.warnings?.length ? 'flagged' : 'clean'}`}>
      {job.verdict === 'violation' || job.warnings?.length ? <ShieldAlert/> : <ShieldCheck/>}
      <div><strong>{t(job.verdict === 'violation' ? 'Обнаружены признаки возможных нарушений' : 'Нарушение не обнаружено')}</strong>
        <p>{t(job.verdict === 'violation' ? 'Проверьте каждое событие по кадрам и таймкоду.' : job.warnings?.length ? 'Результат получен с ограничениями и требует ручной проверки.' : 'В доступной записи не выявлено признаков возможных нарушений.')}</p></div>
      {job.verdict === 'violation' && <span className="review-count">{job.violations.length}</span>}
    </div>}
    <div className="review-grid">
      <div className="review-player-column">
        <div className="review-player">
          <div className="review-player-label"><FileVideo size={14}/>{t('Исходная запись')}<span>{t('Локальный контур')}</span></div>
          <StoredVideo id={job.job_id} localURL={localURL} videoRef={video} autoLoad onLoadedMetadata={onReady} onTimeUpdate={() => setPosition(video.current?.currentTime ?? 0)} />
        </div>
        <div className="event-timeline">
          <div className="flex items-center justify-between gap-3"><h3>{t('Таймлайн событий')}</h3><span className="tabular-nums">{timecode(position)} / {duration ? timecode(duration) : '—'}</span></div>
          <div className="timeline-track">
            <input type="range" min={0} max={duration || 1} step="0.1" value={Math.min(position, duration || 1)} disabled={!duration} aria-label={t('Позиция видео')} aria-valuetext={timecode(position)} onChange={e => seek(Number(e.target.value))}/>
            {duration > 0 && job.violations.map((v, i) => v.timecode_sec <= duration && v.timecode_sec >= 0 && <button key={i} className={`timeline-marker ${i === selected ? 'selected' : ''}`} style={{ left: `${Math.max(1, Math.min(99, v.timecode_sec / duration * 100))}%` }} aria-label={`${t('Перейти к')} ${timecode(v.timecode_sec)} · ${v.type}`} aria-pressed={selected === i} title={`${timecode(v.timecode_sec)} · ${v.type}`} onClick={() => choose(i)}><span>{i + 1}</span></button>)}
          </div>
          <p>{t(duration ? 'Выберите отметку, чтобы перейти к событию.' : 'Таймлайн появится после загрузки видео.')}</p>
          {seekError && <p role="status" className="text-warning-foreground">{t('Таймкод находится за пределами записи. Проверьте кадр-доказательство.')}</p>}
        </div>
        <p className="review-human-note"><ShieldCheck size={16}/>{t('Результат — подсказка для проверки человеком, а не окончательное заключение.')}</p>
      </div>
      <aside className="review-events" aria-label={t('События')}>
        <div className="review-events-heading"><h3>{t('События')}</h3><span>{job.status === 'done' ? job.violations.length : '—'}</span></div>
        {job.violations.length > 0 ? <>
          <div className="event-list">{job.violations.map((v, i) => <button key={i} aria-pressed={selected === i} onClick={() => choose(i)} className={`event-choice ${selected === i ? 'selected' : ''}`}>
            <span className="event-index">{String(i + 1).padStart(2, '0')}</span><span className="min-w-0 flex-1">{v.type}</span><span className="event-time"><Play size={11}/>{timecode(v.timecode_sec)}</span>
          </button>)}</div>
          {event && <article className="event-detail" aria-label={event.type}>
            <h4>{t('Что видно на видео')}</h4><p>{event.description}</p>
            <h4>{t(job.demo ? 'Тестовый кадр' : 'Кадр-доказательство')}</h4>
            <div className="event-evidence">{event.evidence_frames.map((frame, i) => {
              const src = `data:${frame.image_base64.startsWith('iVBOR') ? 'image/png' : 'image/jpeg'};base64,${frame.image_base64}`;
              const label = `${t(job.demo ? 'Тестовый кадр' : 'Кадр-доказательство')} · ${timecode(frame.timestamp_sec)}`;
              return <figure key={i}><button aria-label={`${t('Увеличить кадр')} ${timecode(frame.timestamp_sec)}`} onClick={() => setExpanded({ src, label })}><img src={src} alt={label}/><Maximize2 size={16}/></button><figcaption><button onClick={() => seek(frame.timestamp_sec)}>{timecode(frame.timestamp_sec)}</button></figcaption></figure>;
            })}</div>
            {!event.evidence_frames.length && <p className="text-muted-foreground">{t('Кадры не предоставлены анализатором.')}</p>}
            <div className="event-rule"><ShieldCheck size={17}/><div><h4>{t('Основание для проверки')}</h4>{event.rule ? <><p>{event.rule.title} · {event.rule.section}</p>{/^https?:\/\//.test(event.rule.url) && <a href={event.rule.url} target="_blank" rel="noreferrer">{t('Открыть источник ↗')}</a>}</> : <p>{t('Нормативный источник не предоставлен анализатором.')}</p>}</div></div>
          </article>}
        </> : <div className="review-events-empty">{active ? <Clock3/> : job.status === 'failed' ? <ShieldAlert/> : <ShieldCheck/>}<strong>{t(active ? 'События появятся после анализа' : job.status === 'failed' ? 'Результат недоступен' : 'Нет событий для проверки')}</strong><p>{t(active ? 'Статус обновляется автоматически.' : 'Вы можете просмотреть исходную запись.')}</p></div>}
      </aside>
    </div>
    <Dialog.Root open={!!expanded} onOpenChange={open => { if (!open) setExpanded(null); }}><Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm"/>
      <Dialog.Content className="evidence-dialog"><Dialog.Title>{expanded?.label}</Dialog.Title><Dialog.Description className="sr-only">{t('Кадр для ручной проверки события')}</Dialog.Description><Dialog.Close aria-label={t('Закрыть')}><X/></Dialog.Close>{expanded && <img src={expanded.src} alt={expanded.label}/>}</Dialog.Content>
    </Dialog.Portal></Dialog.Root>
  </section>;
}

function ProcessingState({ status }: { status: Job['status'] }) {
  const { t } = useWorkspace();
  return <div className="processing-state" role="status" aria-live="polite">
    <img src="/brand/cashguard-forte.svg" alt="" className="processing-mark"/>
    <div><strong>{t(status === 'queued' ? 'Видео ожидает своей очереди.' : 'Идёт анализ видео. Это может занять несколько минут.')}</strong>
      <ol className="processing-steps"><li className="complete"><Check size={13}/>{t('Видео получено')}</li><li className={status === 'processing' ? 'current' : ''}>{t('Анализ событий')}</li><li>{t('Результат проверки')}</li></ol>
    </div>
  </div>;
}

export function FirstReview({ onStart }: { onStart: () => void }) {
  const { t } = useWorkspace();
  return <section className="first-review" aria-labelledby="first-review-title">
    <div className="first-review-intro"><span className="first-review-icon"><UploadIcon/></span><div><p className="cashguard-eyebrow">{t('ОТ ВИДЕО К РЕШЕНИЮ')}</p><h2 id="first-review-title">{t('Начните первую проверку')}</h2><p>{t('Загрузите запись. CashGuard поможет найти события, которые требуют внимания.')}</p></div><Button onClick={onStart}><Plus size={16}/>{t('Загрузить видео')}</Button></div>
    <ol className="first-review-steps">{[['01','Загрузите запись','Выберите видео работы кассира.'],['02','Дождитесь анализа','Статус обновляется автоматически.'],['03','Проверьте результат','Изучите события, таймкоды и доказательства.']].map(([n,title,description]) => <li key={n}><span>{n}</span><div><h3>{t(title)}</h3><p>{t(description)}</p></div></li>)}</ol>
  </section>;
}
function UploadIcon() { return <FileVideo size={28}/>; }
