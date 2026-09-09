from sklearn.preprocessing import RobustScaler
from ninflam.core.registry import register

@register("scaler", "robust")
class RobustScalerFactory:
    def build(self):
        return RobustScaler()
