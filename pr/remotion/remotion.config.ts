import { Config } from '@remotion/cli/config'
// @ts-expect-error plain esm helper
import { webpackOverride } from './webpack-override.mjs'

Config.setVideoImageFormat('jpeg')
Config.overrideWebpackConfig(webpackOverride)
