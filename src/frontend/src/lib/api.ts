export type Violation={type:string;timecode_sec:number;description:string;evidence_frames:{timestamp_sec:number;image_base64:string}[];rule?:{title:string;section:string;url:string}};
export type Job={job_id:string;status:'queued'|'processing'|'done'|'failed';verdict?:'violation'|'no_violation';violations:Violation[];warnings?:string[];filename:string;created_at:string;error?:string;demo:boolean};
export async function api<T>(path:string,key:string,options:RequestInit={}):Promise<T>{
 let r:Response;try{r=await fetch('/api'+path,{...options,headers:{...options.headers,...(key?{'X-API-Key':key}:{})}})}catch{throw new ApiError('network_error',0)}
 if(r.status===204)return undefined as T;
 let data;try{data=await r.json()}catch{throw new ApiError('server_error',r.status)}
 if(!r.ok)throw new ApiError(data.error||'server_error',r.status);return data;
}
export const timecode=(n:number)=>`${Math.floor(n/60).toString().padStart(2,'0')}:${Math.floor(n%60).toString().padStart(2,'0')}`;

export class ApiError extends Error {constructor(message:string,public status:number){super(message)}}
