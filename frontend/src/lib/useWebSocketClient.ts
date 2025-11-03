import { useEffect, useRef, useState } from "react";

export function useWebSocketClient(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    const handleOpen = () => setIsOpen(true);
    const handleClose = () => setIsOpen(false);

    ws.addEventListener("open", handleOpen);
    ws.addEventListener("close", handleClose);

    return () => {
      ws.removeEventListener("open", handleOpen);
      ws.removeEventListener("close", handleClose);
      ws.close();
      wsRef.current = null;
    };
  }, [url]);

  const sendJson = (data: unknown) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify(data));
  };

  const subscribe = (onMessage: (event: MessageEvent<string>) => void) => {
    if (!wsRef.current) return () => {};
    const handler = (evt: MessageEvent) => {
      if (typeof evt.data === "string") {
        onMessage(evt as MessageEvent<string>);
      }
    };
    wsRef.current.addEventListener("message", handler as EventListener);
    return () => wsRef.current?.removeEventListener("message", handler as EventListener);
  };

  return { isOpen, sendJson, subscribe };
}


