http://neuralprophet.com/changes-from-prophet/

# 相较于Prophet的改进

NeuralProphet与原来的Prophet相比，增加了一些功能。它们如下。

- 使用PyTorch作为后端进行优化的梯度下降法。
- 使用AR-Net对时间序列的自相关进行建模。
- 使用seepearate前馈神经网络对滞后回归者进行建模。
- 可配置的FFNNs非线性深层。
- 可调整到特定的预测范围（大于1）。
- 自定义损失和指标。

由于代码的模块化和PyTorch支持的可扩展性，任何可通过梯度下降训练的组件都可以作为一个模块添加到NeuralProphet中。使用PyTorch作为后端，与原来使用Stan作为后端的Prophet相比，使得建模过程更快。