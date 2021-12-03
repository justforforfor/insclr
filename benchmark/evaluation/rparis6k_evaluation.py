from .image_search_evaluation import ImageSearchEvaluator


class RParis6kEvaluator(ImageSearchEvaluator):
    def __init__(self, root, img_size, scales, revisitop1m_features=None):
        super().__init__("rparis6k", root, img_size, scales, revisitop1m_features)
