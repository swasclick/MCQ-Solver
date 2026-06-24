from transformers import pipeline

class ZeroShotSolver:
    
    def __init__(self,model="facebook/bart-large-mnli", device_id=-1):
        self.model = model
        self.device_id = device_id
        
    def solve(self, text, labels):
        classifier = pipeline(
            "zero-shot-classification",
            model = self.model,
            device=self.device_id
        )
        result = classifier(text, labels)
        return result
        