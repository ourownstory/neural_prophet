```mermaid
flowchart TD
    BaseComponent --> Trend
    Trend --> LinearTrend
    LinearTrend --> GlobalLinearTrend
    LinearTrend --> LocalLinearTrend
    Trend --> StaticTrend
    Trend --> PiecewiseLinearTrend
    PiecewiseLinearTrend --> GlobalPiecewiseLinearTrend
    PiecewiseLinearTrend --> LocalPiecewiseLinearTrend
```