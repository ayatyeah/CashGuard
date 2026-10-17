import { useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { KeyRound, Loader2, X } from 'lucide-react';
import { Button } from './button';
import { Input } from './input';
import { Label } from './label';
import { api } from '@/lib/api';
import { useWorkspace } from '@/lib/workspace';

export function AccountDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t, token, connect, invalidate } = useWorkspace();
  const [key, setKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  function close() { setKey(''); setError(''); onClose(); }
  async function submit(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError('');
    try { const value = key.trim(); await api('/jobs', value); connect(value); close(); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return (
    <Dialog.Root open={open} onOpenChange={(v) => { if (!v && !busy) close(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 flex max-h-[90vh] w-[calc(100vw-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 flex-col gap-4 overflow-auto rounded-xl border bg-card p-6 text-card-foreground shadow-xl">
          <Dialog.Close disabled={busy} className="absolute right-4 top-4 rounded-sm text-muted-foreground opacity-70 hover:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" aria-label={t('Закрыть')}>
            <X className="size-4" />
          </Dialog.Close>
          <span className="grid size-12 place-items-center rounded-xl bg-accent text-accent-foreground"><KeyRound className="size-6" /></span>
          <div className="space-y-1.5">
            <Dialog.Title className="text-lg font-semibold">{t('Настройки подключения')}</Dialog.Title>
            <Dialog.Description className="text-sm text-muted-foreground">{t('Введите API-ключ сервера. Видео и результаты хранятся в общем рабочем пространстве.')}</Dialog.Description>
          </div>
          {token && <p className="rounded-md bg-success-soft px-3 py-2 text-sm text-success-foreground">{t('API-ключ подключён')}</p>}
          <form onSubmit={submit} className="flex flex-col gap-3">
            <Label htmlFor="api-key">API key</Label>
            <Input id="api-key" type="password" autoComplete="off" required value={key} onChange={(e) => setKey(e.target.value)} disabled={busy} />
            <Button type="submit" className="gap-2" disabled={busy || !key.trim()}>
              {busy && <Loader2 className="size-4 animate-spin" />}{t('Подключить')}
            </Button>
          </form>
          {token && <Button variant="outline" disabled={busy} onClick={() => { invalidate(); close(); }}>{t('Отключить')}</Button>}
          {error && <p role="alert" className="rounded-md bg-destructive-soft px-3 py-2 text-sm text-destructive">{t(error)}</p>}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
