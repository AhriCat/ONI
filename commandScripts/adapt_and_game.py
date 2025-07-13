def adaptive_game_agent(oni: OniMicro, n_steps: int = 10_000):
    """
    Uses OniMicro's internal recurrent DQN and spatial memory
    to learn policies without explicit game API access
    (screen capture + key/mouse actuator).
    """
    oni.start_processing_feed()        # start vision thread
    oni.start_processing_audio()       # optional audio
    hidden = None

    for t in range(n_steps):
        screen_state = oni.capture_screen()  # tensor (1,C,H,W)
        # trivial vision embed – replace with oni.vision_module for speed
        img_vec = screen_state.mean(dim=[2, 3]).cpu().numpy()
        state_vec = oni.get_state(img_vec)

        action_id, hidden = oni.choose_action(state_vec, hidden)
        # map discrete action index → actual keystrokes/mouse movement
        oni.use_mouse_and_keyboard(_minecraft_action_map(action_id))

        reward = _compute_game_reward()  # user‑defined
        next_state = oni.get_state(img_vec)  # simplistic; real impl diff.

        oni.train_rl(state_vec, action_id, reward, next_state, done=False)

        if t % 300 == 0:
            print(f"[{t}] ε={oni.exploration_rate:.3f}")

    oni.stop_processing_feed()
    oni.stop_processing_audio()

oni = OniMicro(
    tokenizer=tokenizer,
    input_dim=896,
    hidden_dim=896,
    output_dim=896,
    nhead=8,
    num_layers=20,
    exec_func=exec_func,
    state_size=256,
    action_size=20,
)

print(adaptive_game_agent(oni, n_steps = 10_000))
