from Oni.py import OniMicro

#use import weights function if you want pretrained on the oni variable
oni = OniMicro()
# Core agent loop for open-ended cognition
goals = []

while oni.awake():  # persistent loop: could be tied to energy, task queue, etc.
    
    # Step 1: Reflect on current priorities
    reflection = oni.nlp.generate("What are my current goals?")
    
    # Step 2: Update internal goal state
    if not goals:
        goals = [reflection]
    else:
        goals.append(reflection)

    # Step 3: Explore or act based on the goal trajectory
    oni.explore(goals)

    # Optional: prune or decay old goals
    goals = oni.exec_func(goals)
  # e.g., based on relevance, energy, novelty
   # oni.robotics_controller(goals)
   # oni.explore(goals)
   # oni.generate_response(goals)
