from .image_search_evaluation import ImageSearchEvaluator


class ROxford5kEvaluator(ImageSearchEvaluator):
    def __init__(self, root, img_size, scales, revisitop1m_features=None):
        super().__init__("roxford5k", root, img_size, scales, revisitop1m_features)
