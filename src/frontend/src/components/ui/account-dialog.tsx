import {useState} from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import {Loader2,KeyRound,X} from 'lucide-react';
import {Button} from './button';
import {api} from '@/lib/api';
import {useWorkspace} from '@/lib/workspace';
export function AccountDialog({open,onClose}:{open:boolean;onClose:()=>void}){
 const {t,token,connect,invalidate}=useWorkspace();const [key,setKey]=useState('');const [busy,setBusy]=useState(false);const [error,setError]=useState('');
 function close(){setKey('');setError('');onClose()}
 async function submit(e:React.FormEvent){e.preventDefault();setBusy(true);setError('');try{const value=key.trim();await api('/jobs',value);connect(value);close()}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 return <Dialog.Root open={open} onOpenChange={v=>{if(!v&&!busy)close()}}><Dialog.Portal><Dialog.Overlay className="account-overlay"/><Dialog.Content className="account-dialog"><Dialog.Close disabled={busy} className="icon-button modal-close" aria-label={t('Закрыть')}><X size={20}/></Dialog.Close><span className="upload-icon"><KeyRound size={26}/></span><Dialog.Title>{t('Настройки подключения')}</Dialog.Title><Dialog.Description>{t('Введите API-ключ сервера. Видео и результаты хранятся в общем рабочем пространстве.')}</Dialog.Description>{token&&<p>{t('API-ключ подключён')}</p>}<form onSubmit={submit}><label htmlFor="api-key">API key</label><input id="api-key" type="password" autoComplete="off" required value={key} onChange={e=>setKey(e.target.value)} disabled={busy}/><Button type="submit" disabled={busy||!key.trim()}>{busy&&<Loader2 className="spin" size={16}/>} {t('Подключить')}</Button></form>{token&&<Button variant="outline" disabled={busy} onClick={()=>{invalidate();close()}}>{t('Отключить')}</Button>}{error&&<p className="error-banner" role="alert">{t(error)}</p>}</Dialog.Content></Dialog.Portal></Dialog.Root>
}
