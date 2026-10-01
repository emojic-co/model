import { createRoot } from 'react-dom/client'
import { App } from './App'
import { LangProvider } from './i18n'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <LangProvider>
    <App />
  </LangProvider>,
)
