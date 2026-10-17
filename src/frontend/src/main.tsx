import React from 'react';
import ReactDOM from 'react-dom/client';
import App1 from '@/components/ui/app-1';
import {WorkspaceProvider,useWorkspace} from '@/lib/workspace';
import './index.css';
function SessionApp(){const {token}=useWorkspace();return <App1 key={token}/>;}
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><WorkspaceProvider><SessionApp/></WorkspaceProvider></React.StrictMode>);
