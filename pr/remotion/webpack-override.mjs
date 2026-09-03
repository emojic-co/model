import path from 'node:path'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const pkgDir = (name) => path.dirname(require.resolve(`${name}/package.json`))

export const webpackOverride = (config) => ({
  ...config,
  resolve: {
    ...config.resolve,
    alias: {
      ...(config.resolve?.alias ?? {}),
      react: pkgDir('react'),
      'react-dom': pkgDir('react-dom'),
    },
  },
})
