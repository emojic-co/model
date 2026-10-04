import { Composition } from 'remotion'
import { FPS, HEIGHT, Showcase, WIDTH } from './Showcase'
import { ShowcaseConfig, durationInFrames } from './config'

// One composition per configs/*.json (id = file name); drop a new JSON in configs/ to get a new video.
const context = (require as any).context('../configs', false, /\.json$/)
const CONFIGS: [string, ShowcaseConfig][] = context
  .keys()
  .map((k: string) => [k.replace(/^\.\/|\.json$/g, ''), context(k)])

export const RemotionRoot: React.FC = () => (
  <>
    {CONFIGS.map(([id, config]) => (
      <Composition
        key={id}
        id={id}
        component={Showcase}
        defaultProps={{ config }}
        durationInFrames={durationInFrames(config, FPS)}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    ))}
  </>
)
