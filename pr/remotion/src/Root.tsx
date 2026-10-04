import { Composition } from 'remotion'
import { DURATION_FRAMES, FPS, HEIGHT, Showcase, WIDTH } from './Showcase'

export const RemotionRoot: React.FC = () => (
  <Composition
    id="showcase"
    component={Showcase}
    durationInFrames={DURATION_FRAMES}
    fps={FPS}
    width={WIDTH}
    height={HEIGHT}
  />
)
