# module_fitness.py

class ModuleFitnessEvaluator:
    def __init__(self, registry, test_loader, device="cuda"):
        self.registry = registry  # Dict of module_name -> module_class
        self.test_loader = test_loader  # Dict of module_name -> test_fn
        self.device = device

    def evaluate_module(self, module_name: str):
        module_class = self.registry[module_name]
        test_fn = self.test_loader[module_name]
        
        model = module_class().to(self.device)
        result = test_fn(model)

        report = {
            "module": module_name,
            "param_count": sum(p.numel() for p in model.parameters()),
            "success": result["success"],
            "details": result["details"],
            "inference_time": result["inference_time"],
            "recommended_size": self.recommend_size(result, model),
        }

        return report

    def recommend_size(self, result, model):
        if not result["success"]:
            return "increase"
        if result["inference_time"] > result["target_time"]:
            return "prune or compress"
        return "ok"
