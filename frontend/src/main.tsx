import React, { FormEvent, useState } from 'react'
import { createRoot } from 'react-dom/client'

export interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ role, content, sources }) => {
  const isAssistant = role === 'assistant';

  return (
    <div
      className={`max-w-[85%] sm:max-w-[75%] rounded-[1.5rem] p-5 my-3 shadow-sm ${
        isAssistant 
          ? 'bg-white mr-auto text-brick-navy border border-gray-100'
          : 'bg-brick-navy ml-auto text-white'
      }`}
    >
      <div className="whitespace-pre-wrap leading-relaxed text-[15px]">
        {content}
      </div>

      {isAssistant && sources && sources.length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-100">
          <h4 className="text-[10px] font-bold text-gray-400 mb-2 tracking-widest uppercase">
            Källor:
          </h4>
          <div className="flex flex-wrap gap-2 text-sm">
            {sources.map((source, index) => (
              <span
                key={index}
                title={source}
                className="inline-flex items-center px-3 py-1 rounded-full text-[11px] font-semibold bg-[#F5F7FA] text-brick-navy hover:bg-[#E2E8F0] transition-colors border border-gray-200 shadow-sm whitespace-nowrap cursor-default"
              >
                <span className="mr-1.5 opacity-70">📄</span>
                <span className="truncate max-w-[200px] sm:max-w-xs block">
                  {source}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

function App() {
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string; sources?: string[] }[]>([])

  console.log("API URL at build time:", (import.meta as any).env.VITE_API_URL)
  const BASE_URL = (import.meta as any).env.VITE_API_URL || "https://foolish-stork-kth-57b627c3.koyeb.app";
  const endpoint = `${BASE_URL.replace(/\/$/, '')}/chat`;

  const send = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || thinking) return
    const history = [...messages]
    setInput('')
    setThinking(true)
    setMessages([...history, { role: 'user', content: input.trim() }])
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input.trim(), history }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMsg = "";
      let sourcesList: string[] = [];

      setMessages((m) => [...m, { role: 'assistant', content: "", sources: [] }]);
      setThinking(false);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        assistantMsg += chunk;

        if (assistantMsg.includes("__SOURCES_METADATA__")) {
          const parts = assistantMsg.split("__SOURCES_METADATA__");
          assistantMsg = parts[0];
          try {
            sourcesList = JSON.parse(parts[1]);
          } catch (e) {}
        }

        setMessages((m) => {
          const newM = [...m];
          newM[newM.length - 1] = { role: 'assistant', content: assistantMsg, sources: sourcesList };
          return newM;
        });
      }
    } catch {
      setMessages((m) => [...m, { role: 'assistant', content: "Nätverksfel vid kontakt med servern." }])
    }
    setThinking(false)
  }

  return (
    <div className="w-full">
      {/* Navigation removed */}

      {/* Hero Section */}
      <header className="px-8 pt-20 pb-16 max-w-7xl mx-auto w-full flex flex-col items-center text-center">
        {/* Badge */}
        <div className="bg-transparent border border-gray-300/80 text-brick-navy px-4 py-1.5 rounded-full text-[13px] font-bold uppercase tracking-wider mb-6 flex items-center gap-2 bg-white/50 backdrop-blur-sm">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brick-coral opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-brick-coral"></span>
          </span>
          RAG-Agent Beta
        </div>
        
        <h1 className="font-display text-5xl md:text-7xl font-bold tracking-tight leading-[1.05]">
          Bygg med <span className="text-brick-coral relative inline-block">Impact<svg className="absolute -bottom-2 left-0 w-full h-3 text-brick-coral/30" viewBox="0 0 100 20" preserveAspectRatio="none"><path d="M0 10 Q 50 20 100 10" stroke="currentColor" strokeWidth="8" fill="transparent"/></svg></span>
        </h1>
        
        <h2 className="font-serif text-lg sm:text-xl text-brick-navy/70 mt-6 mb-12 max-w-2xl leading-relaxed">
          Samhällsbyggarbranschens AI-agentiska plattform för projektanalys, anbud och inköp.
        </h2>
      </header>

      {/* RAG Chat / Tool Interface */}
      <main className="max-w-4xl mx-auto w-full px-4 pb-24">
        <div className="bg-[#F8F9FA] border border-gray-200/60 rounded-3xl p-4 sm:p-6 shadow-[0px_4px_24px_rgba(0,0,0,0.02)] relative">
          {/* Chat Window */}
          <div className="h-[450px] overflow-y-auto mb-6 pr-2 custom-scrollbar flex flex-col gap-2">
            {messages.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center opacity-50 px-4">
                <div className="w-16 h-16 mb-4 rounded-full bg-gray-200 flex items-center justify-center">
                  <span className="text-2xl">🤖</span>
                </div>
                <h3 className="font-display font-semibold text-xl mb-2">Agenten är redo</h3>
                <p className="text-sm max-w-sm">Ställ tekniska frågor relaterade till AMA eller mängdförteckningen nedan.</p>
              </div>
            ) : (
              messages.map((message, i) => (
                <ChatMessage 
                  key={i} 
                  role={message.role} 
                  content={message.content} 
                  sources={message.sources}
                />
              ))
            )}
            {thinking && (
              <div className="flex items-center gap-3 text-brick-navy/60 text-sm font-medium ml-4 my-2 opacity-70">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-brick-coral rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-brick-coral rounded-full animate-bounce [animation-delay:0.2s]"></div>
                  <div className="w-2 h-2 bg-brick-coral rounded-full animate-bounce [animation-delay:0.4s]"></div>
                </div>
                Analyserar underlag...
              </div>
            )}
          </div>

          {/* Input Form */}
          <form onSubmit={send} className="relative z-10 w-full">
            <div className="relative flex items-center bg-white rounded-full border border-gray-300 shadow-sm focus-within:ring-2 focus-within:ring-brick-coral/20 focus-within:border-brick-coral transition-all overflow-hidden p-1.5">
              <input 
                autoFocus
                value={input} 
                onChange={(e) => setInput(e.target.value)} 
                placeholder="Ex. Vilka nyckelroller krävs i anbudsformuläret?" 
                className="flex-1 bg-transparent px-4 py-3 text-base text-brick-navy placeholder-gray-400 focus:outline-none"
                disabled={thinking}
              />
              <button 
                type="submit"
                disabled={thinking || !input.trim()}
                className="bg-brick-coral text-white p-3 md:px-6 rounded-full font-semibold transition-all hover:bg-opacity-90 shadow-md disabled:opacity-50 disabled:hover:-translate-y-0"
              >
                <span className="hidden sm:inline">Skicka</span>
                <span className="sm:hidden">→</span>
              </button>
            </div>
          </form>
        </div>
      </main>
      


      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background-color: #CBD5E1; border-radius: 20px; }
      `}} />
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
