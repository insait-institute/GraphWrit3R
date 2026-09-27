"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.multi_teacher = isinstance(self.cfg, dict)
        if self.multi_teacher:
            self.criteria = {
                name: [LOSSES.build(cfg=loss_cfg) for loss_cfg in loss_cfgs]
                for name, loss_cfgs in self.cfg.items()
            }
        else:
            self.criteria = []
            for loss_cfg in self.cfg:
                self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target, return_dict=False, **kwargs):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred

        if not self.multi_teacher:
            loss = 0
            for c in self.criteria:
                loss += c(pred, target, **kwargs)
            return loss

        loss = 0
        loss_dict = {}
        for name, criterias in self.criteria.items():
            if isinstance(pred, dict):
                pred_item = pred[name]
            else:
                raise TypeError(
                    "Predictions must be a dict when using multi-teacher criteria."
                )
            if isinstance(target, dict):
                target_item = target[name]
            else:
                raise TypeError(
                    "Targets must be a dict when using multi-teacher criteria."
                )

            teacher_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    if name in value:
                        teacher_kwargs[key] = value[name]
                else:
                    teacher_kwargs[key] = value

            teacher_loss = 0
            for c in criterias:
                teacher_loss += c(pred_item, target_item, **teacher_kwargs)
            loss_dict[name] = teacher_loss
            loss += teacher_loss

        if return_dict:
            loss_dict["total"] = loss
            return loss_dict
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
