export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string; // markdown supported
};

export type SliderSpec = {
  id: string;
  label: string;
  min: number; // e.g., -1
  max: number; // e.g., 1
  step: number; // e.g., 0.01
  value: number; // centered default, e.g., 0
};

export type AppState = {
  sliders: SliderSpec[];
};

// Outgoing message payload shape (proposal)
export type SendChatPayload = {
  type: "chat";
  message: string;
  state: AppState;
};

export type SendIdeaPayload = {
  type: "idea";
  idea: string;
  state: AppState;
};

// Incoming server events (proposal)
export type ServerEvent =
  | { type: "assistant_token"; token: string }
  | { type: "assistant_done" }
  | { type: "system"; content: string }
  | { type: "add_slider"; slider: SliderSpec };


