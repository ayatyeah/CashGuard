import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import { Loader2, Play } from 'lucide-react';
import { Button } from './button';
import { useWorkspace } from '@/lib/workspace';

export function StoredVideo({ id, localURL, videoRef, autoLoad = false, onLoadedMetadata, onTimeUpdate }: {
  id: string; localURL?: string; videoRef: RefObject<HTMLVideoElement>; autoLoad?: boolean;
  onLoadedMetadata?: () => void; onTimeUpdate?: () => void;
}) {
  const { token, t } = useWorkspace();
  const [url, setURL] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const controller = useRef<AbortController>();
  const objectURL = useRef('');
  const load = useCallback(async () => {
    controller.current?.abort();
    setLoading(true); setError('');
    const abort = new AbortController(); controller.current = abort;
    try {
      const response = await fetch(`/api/jobs/${id}/video`, { headers: { 'X-API-Key': token }, signal: abort.signal });
      if (!response.ok) throw new Error('Ошибка чтения видео');
      const blob = await response.blob();
      if (abort.signal.aborted) return;
      if (objectURL.current) URL.revokeObjectURL(objectURL.current);
      const next = URL.createObjectURL(blob); objectURL.current = next; setURL(next);
    } catch (e) { if (!abort.signal.aborted) setError((e as Error).message); }
    finally { if (!abort.signal.aborted) setLoading(false); }
  }, [id, token]);
  useEffect(() => {
    if (autoLoad && !localURL) void load();
    return () => { controller.current?.abort(); if (objectURL.current) URL.revokeObjectURL(objectURL.current); };
  }, [autoLoad, localURL, load]);
  return <div>
    {localURL || url ? <video ref={videoRef} className="max-h-[520px] w-full rounded-lg bg-black" src={localURL || url} controls playsInline preload="metadata" onLoadedMetadata={onLoadedMetadata} onDurationChange={onLoadedMetadata} onTimeUpdate={onTimeUpdate} onError={() => setError('Не удалось воспроизвести видео в браузере.')} />
      : <div className="video-loading"><Button variant="outline" size="sm" className="gap-2" disabled={loading} onClick={load}>
        {loading ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}{t(loading ? 'Загрузка видео…' : 'Открыть видео')}
      </Button></div>}
    {error && <p role="alert" className="p-3 text-sm text-destructive">{t(error)}</p>}
  </div>;
}
