import { createRoot } from 'react-dom/client'
import { App } from './App'
import { restoreRedirectedPath } from './nav'
import './styles.css'

restoreRedirectedPath()

createRoot(document.getElementById('root')).render(<App />)
