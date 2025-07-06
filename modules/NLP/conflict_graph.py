
class PrincipleConflictGraph:
    """Tracks and analyzes conflicts between principles in a graph structure."""
    
    def __init__(self, threshold: float = 0.5):
        self.conflict_graph = defaultdict(set)
        self.conflict_scores = {}
        self.threshold = threshold
        self.conflict_history = []
        
    def add_conflict(self, principle_a: int, principle_b: int, score: float, context: torch.Tensor) -> None:
        """Add a conflict between two principles with context."""
        if score > self.threshold:
            key = (min(principle_a, principle_b), max(principle_a, principle_b))
            self.conflict_graph[principle_a].add(principle_b)
            self.conflict_graph[principle_b].add(principle_a)
            self.conflict_scores[key] = max(score, self.conflict_scores.get(key, 0))
            
            # Record conflict with context
            self.conflict_history.append({
                'principles': key,
                'score': score,
                'context_embedding': context.mean(0).detach().cpu().numpy(),
                'timestamp': len(self.conflict_history)
            })
    
    def get_conflicts(self, principle: int) -> List[Tuple[int, float]]:
        """Get all principles that conflict with the given principle."""
        conflicts = []
        for other in self.conflict_graph[principle]:
            key = (min(principle, other), max(principle, other))
            conflicts.append((other, self.conflict_scores[key]))
        return sorted(conflicts, key=lambda x: x[1], reverse=True)
    
    def get_all_conflicts(self) -> List[Tuple[int, int, float]]:
        """Get all conflicts in the graph."""
        conflicts = []
        for key, score in self.conflict_scores.items():
            conflicts.append((key[0], key[1], score))
        return sorted(conflicts, key=lambda x: x[2], reverse=True)
    
    def get_most_conflicted_principles(self, top_k: int = 5) -> List[Tuple[int, int]]:
        """Get the principles involved in the most conflicts."""
        principle_conflict_count = defaultdict(int)
        for p1, p2, _ in self.get_all_conflicts():
            principle_conflict_count[p1] += 1
            principle_conflict_count[p2] += 1
        
        return sorted(principle_conflict_count.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def analyze_conflict_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the conflict graph."""
        if not self.conflict_history:
            return {"patterns": [], "clusters": 0}
        
        # Convert conflict history to numpy array for analysis
        contexts = np.array([c['context_embedding'] for c in self.conflict_history])
        
        # Simple clustering to find patterns (in a real implementation, use a proper clustering algorithm)
        from sklearn.cluster import KMeans
        if len(contexts) > 5:  # Need enough samples for meaningful clustering
            n_clusters = min(5, len(contexts) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(contexts)
            
            # Analyze each cluster
            clusters = []
            for i in range(n_clusters):
                cluster_indices = np.where(kmeans.labels_ == i)[0]
                cluster_conflicts = [self.conflict_history[idx] for idx in cluster_indices]
                
                # Find common principles in this cluster
                principles_count = defaultdict(int)
                for conflict in cluster_conflicts:
                    principles_count[conflict['principles'][0]] += 1
                    principles_count[conflict['principles'][1]] += 1
                
                top_principles = sorted(principles_count.items(), key=lambda x: x[1], reverse=True)[:3]
                
                clusters.append({
                    "size": len(cluster_indices),
                    "top_principles": top_principles,
                    "avg_score": np.mean([c['score'] for c in cluster_conflicts])
                })
            
            return {
                "patterns": clusters,
                "clusters": n_clusters
            }
        
        return {"patterns": [], "clusters": 0}
