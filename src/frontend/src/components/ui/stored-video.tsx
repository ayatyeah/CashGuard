import { useEffect, useRef, useState, type RefObject } from 'react';
import { Loader2, Play } from 'lucide-react';
import { Button } from './button';
import { useWorkspace } from '@/lib/workspace';

export function StoredVideo({ id, localURL, videoRef }: { id: string; localURL?: string; videoRef: RefObject<HTMLVideoElement> }) {
  const { token, t } = useWorkspace();
  const [url, setURL] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const controller = useRef<AbortController>();
  const objectURL = useRef('');
  useEffect(() => () => { controller.current?.abort(); if (objectURL.current) URL.revokeObjectURL(objectURL.current); }, []);
  async function load() {
    setLoading(true); setError('');
    const abort = new AbortController(); controller.current = abort;
    try {
      const response = await fetch(`/api/jobs/${id}/video`, { headers: { 'X-API-Key': token }, signal: abort.signal });
      if (!response.ok) throw new Error('Ошибка чтения видео');
      const blob = await response.blob();
      if (abort.signal.aborted) return;
      const next = URL.createObjectURL(blob); objectURL.current = next; setURL(next);
    } catch (e) { if (!abort.signal.aborted) setError((e as Error).message); }
    finally { if (!abort.signal.aborted) setLoading(false); }
  }
  return (
    <div>
      {localURL || url
        ? <video ref={videoRef} className="max-h-[420px] w-full rounded-lg bg-black" src={localURL || url} controls />
        : <Button variant="outline" size="sm" className="gap-2" disabled={loading} onClick={load}>
            {loading ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}{t(loading ? 'Загрузка видео…' : 'Открыть видео')}
          </Button>}
      {error && <p role="alert" className="mt-2 text-sm text-destructive">{t(error)}</p>}
    </div>
  );
}
