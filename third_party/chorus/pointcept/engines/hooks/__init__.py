"""Minimal hook interface used by vendored Pointcept modules."""


class HookBase:
    trainer = None

    def before_train(self):
        pass

    def before_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass

    def before_eval(self):
        pass

