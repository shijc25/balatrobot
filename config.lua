BALATRO_BOT_CONFIG = {
    enabled = true, -- Disables ALL mod functionality if false
    port = '12345', -- Port for the bot to listen on, overwritten by arg[1]
    --dt = 5.0/60.0, -- Tells the game that every update is dt seconds long
    --uncap_fps = true,
    --instant_move = true,
    --disable_vsync = true,
    --disable_card_eval_status_text = true, -- e.g. +10 when scoring a queen
    --disable_chip_easing = true, -- Chips don't tick up/down, they're set directly
    disable_delay = false, -- Disables the delay function, so that the game doesn't wait for a certain time before executing the next action, may cause race conditions
    frame_ratio = 1, -- Draw every 100th frame, set to 1 for normal rendering
}

return BALATRO_BOT_CONFIG