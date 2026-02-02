sequenceDiagram
autonumber
actor ClientA
actor ClientB
participant S as Server

loop Session lifecycle (forced disconnect after 2h -> reconnect)
  Note over ClientA,S: 1) ClientA connects via WSS using URL query params (client name, userId)
  ClientA->>S: WS Connect wss://.../ws?client=clientA&userId=...
  S-->>ClientA: WS Connected + sessionIdA
  ClientA->>ClientA: store sessionIdA

  Note over ClientA,S: 2) ClientA queries ClientB connection status via WS message\nIf connected, store sessionIdB as destination\nIf not, wait for a "ClientB connected" notification
  ClientA->>S: WS Send CHECK_CLIENT_CONNECTED { targetClient: "clientB" }
  alt ClientB already connected
    S-->>ClientA: WS Reply CLIENT_CONNECTED { sessionIdB }
    ClientA->>ClientA: store sessionIdB as destination
  else ClientB not connected yet
    S-->>ClientA: WS Reply CLIENT_NOT_CONNECTED
    ClientA->>S: WS Send SUBSCRIBE_CLIENT_CONNECTED { targetClient: "clientB" }
    Note over ClientA,S: wait until ClientB connects

    ClientB->>S: WS Connect wss://.../ws?client=clientB&userId=...
    S-->>ClientB: WS Connected + sessionIdB
    ClientB->>ClientB: store sessionIdB

    S-->>ClientA: WS Push CLIENT_CONNECTED { sessionIdB }
    ClientA->>ClientA: store sessionIdB as destination
  end

  Note over ClientA,ClientB: 3) After pairing, ClientA sends initialization data to ClientB
  ClientA->>S: WS Send INIT_DATA { to: sessionIdB, payload: ... }
  S-->>ClientB: WS Deliver INIT_DATA

  Note over ClientA,ClientB: 4) Then ClientA and ClientB exchange messages
  par Message exchange (as needed)
    loop as needed
      ClientA->>S: WS Send MESSAGE { to: sessionIdB, payload: ... }
      S-->>ClientB: WS Deliver MESSAGE
      ClientB->>S: WS Send MESSAGE { to: sessionIdA, payload: ... }
      S-->>ClientA: WS Deliver MESSAGE
    end
  and 5) Keep-alive: disconnect after 10 min idle, so send ping about every 10 min
    loop every ~10 min
      ClientA->>S: WS Ping
      S-->>ClientA: Pong (optional)
      ClientB->>S: WS Ping
      S-->>ClientB: Pong (optional)
    end
  end

  Note over ClientA,ClientB,S: 6) Even with pings, connection closes after 2h -> go back to step 1
  S-->>ClientA: WS Close (2h limit)
  S-->>ClientB: WS Close (2h limit)
  ClientA->>ClientA: detect close -> prepare reconnect
  ClientB->>ClientB: detect close -> prepare reconnect
end
