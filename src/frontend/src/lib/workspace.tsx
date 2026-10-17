import {createContext,useContext,useEffect,useState,type ReactNode} from 'react';
import {api,ApiError} from './api';
import {translate,type Language} from './translations';
export type Theme='light'|'dark';
function useWorkspaceState(){
 const [language,setLanguage]=useState<Language>(()=>{const v=localStorage.getItem('cashguard-language');return v==='en'||v==='kk'?v:'ru'});
 const theme:Theme='dark'; // single dark theme, no switch
 const [token,setToken]=useState(()=>sessionStorage.getItem('cashguard-api-key')||'');
 const t=(text:string)=>translate(text,language);
 function invalidate(){sessionStorage.removeItem('cashguard-api-key');setToken('')}
 function connect(value:string){sessionStorage.setItem('cashguard-api-key',value);setToken(value)}
 useEffect(()=>{sessionStorage.removeItem('cashguard-session');let stopped=false;if(token)api('/jobs',token).catch(e=>{if(!stopped&&e instanceof ApiError&&e.status===401)invalidate()});return()=>{stopped=true}},[token]);
 useEffect(()=>{localStorage.removeItem('cashguard-theme');const icon=document.querySelector<HTMLLinkElement>('link[rel="icon"]');if(icon)icon.href='/brand/cashguard-forte.svg'},[]);
 useEffect(()=>{document.documentElement.lang=language;document.title=translate('CashGuard — контроль кассовых операций',language);localStorage.setItem('cashguard-language',language)},[language]);
 async function preferences(nextLanguage:Language){setLanguage(nextLanguage)}
 return {language,theme,token,ready:true,saving:false,settingsError:'',t,preferences,connect,invalidate,locale:language==='kk'?'kk-KZ':language==='en'?'en-US':'ru-RU'};
}
const WorkspaceContext=createContext<ReturnType<typeof useWorkspaceState>|null>(null);
export function WorkspaceProvider({children}:{children:ReactNode}){return <WorkspaceContext.Provider value={useWorkspaceState()}>{children}</WorkspaceContext.Provider>}
export function useWorkspace(){const c=useContext(WorkspaceContext);if(!c)throw new Error('WorkspaceProvider required');return c}
