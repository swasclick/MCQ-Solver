from transformers import pipeline

class Solver:
    
    def __init__(self,model="facebook/bart-large-mnli", device_id=-1):
        self.model = model
        self.device= device_id
        
    def solve(self, text, labels):
        classifier = pipeline(
            "zero-shot-classification",
            model = self.model,
            device=self.device
        )
        text = "Pick the best possible answer: What is Martin Heidegger's view on the relationship between time and human existence? among the listed options."

        labels = ["Martin Heidegger believes that humans exist within a time continuum that is infinite and does not have a defined beginning or end. The relationship to the past involves acknowledging it as a historical era, and the relationship to the future involves creating a world that will endure beyond one's own time."
                  , "Martin Heidegger believes that humans do not exist inside time, but that they are time. The relationship to the past is a present awareness of having been, and the relationship to the future involves anticipating a potential possibility, task, or engagement.", 
                  "Martin Heidegger does not believe in the existence of time or that it has any effect on human consciousness. The relationship to the past and the future is insignificant, and human existence is solely based on the present.", 
                  "Martin Heidegger believes that the relationship between time and human existence is cyclical. The past and present are interconnected and the future is predetermined. Human beings do not have free will.",
                  "Martin Heidegger believes that time is an illusion, and the past, present, and future are all happening simultaneously. Humans exist outside of this illusion and are guided by a higher power."]

        result = classifier(text, labels)

        print(result)
        
Solver().solve('a','b')