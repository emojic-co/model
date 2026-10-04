import path from 'node:path'
import { createRequire } from 'node:module'

const require = createRequire(path.join(process.cwd(), 'package.json'))
const pkgDir = (name) => path.dirname(require.resolve(`${name}/package.json`))
const { DefinePlugin } = require('webpack')

// The web app's modules use Vite idioms (`?raw` imports, import.meta.env); shim them here.
export const webpackOverride = (config) => ({
  ...config,
  module: {
    ...config.module,
    rules: [
      ...(config.module?.rules ?? []),
      { resourceQuery: /raw/, type: 'asset/source' },
      // web/package.json is "type": "module", whose extensionless imports webpack rejects by default.
      { test: /\.jsx?$/, resolve: { fullySpecified: false } },
    ],
  },
  plugins: [
    ...(config.plugins ?? []),
    new DefinePlugin({
      'import.meta.env.BASE_URL': JSON.stringify('/'),
    }),
  ],
  resolve: {
    ...config.resolve,
    alias: {
      ...(config.resolve?.alias ?? {}),
      react: pkgDir('react'),
      'react-dom': pkgDir('react-dom'),
    },
  },
})
