from paddle.optimizer.lr import LRScheduler

"""
    .. code-block:: text
        learning_rate = 0.5
        milestones = [2, 5, 7]
        gamma = [0.2, 0.1, 0.1]
        if epoch < 2:
            learning_rate = 0.5
        elif 2< epoch < 5 :
            learning_rate = 0.5*0.2
        elif 5< epoch < 7 :
             learning_rate = 0.5*0.2*0.1
        elif  epoch > 7 :
             learning_rate = 0.5*0.2*0.1*0.1

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        milestones (tuple|list): List or tuple of each boundaries. Must be increasing.
        gamma (float, optional): List or tuple of each boundaries.  It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``MultiEpochDecay`` instance to schedule learning rate.

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
"""


class MultiEpochDecay(LRScheduler):

    def __init__(self,
                 learning_rate,
                 milestones,
                 gamma,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s." % type(
                    milestones))

        if not all([milestones[i] < milestones[i + 1] for i in range(len(milestones) - 1)]):
            raise ValueError('The elements of milestones must be incremented')

        if not isinstance(gamma, (tuple, list)):
            raise TypeError(
                "The type of 'gamma' in 'MultiEpochDecay' must be 'tuple, list', but received %s." % type(gamma))

        self.milestones = milestones
        self.gamma = gamma
        super(MultiEpochDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch in self.milestones:
            self.last_lr *= self.gamma[self.milestones.index(self.last_epoch)]
        return self.last_lr



if __name__ == '__main__':

    import paddle
    import numpy as np

    # train on default dynamic graph mode
    linear = paddle.nn.Linear(10, 10)
    scheduler = MultiEpochDecay(learning_rate=0.5, milestones=[2, 5, 7], gamma=[0.2, 0.1, 0.1], verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
    for epoch in range(20):
        for batch_id in range(5):
            x = paddle.uniform([10, 10])
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_gradients()
        scheduler.step()  # If you update learning rate each epoch
