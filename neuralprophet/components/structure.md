```mermaid
flowchart TD
    Trend
    BaseComponent --> BaseTrend
    BaseTrend --> LinearTrend
    BaseTrend --> PiecewiseLinearTrend
    PiecewiseLinearTrend --> GlobalPiecewiseLinearTrend
    PiecewiseLinearTrend --> LocalPiecewiseLinearTrend
    BaseTrend --> LogarithmicTrend
```