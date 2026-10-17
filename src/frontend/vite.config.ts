import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
export default defineConfig({plugins:[react()],resolve:{alias:{'@':path.resolve(__dirname,'src')}},server:{proxy:{'/api':process.env.API_PROXY_TARGET || 'http://127.0.0.1:8080'}}});

