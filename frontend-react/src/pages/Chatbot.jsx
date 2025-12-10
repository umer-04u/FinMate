import React, { useState, useRef, useEffect } from 'react';
import { api } from '../services/api'; // We'll add a chat endpoint later
import { Send, Bot, User } from 'lucide-react';

const Chatbot = () => {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Hello! I am your AI Financial Assistant. Ask me anything about your budget or spending habits.' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            // Temporary: echo response or mock until backend is ready
            // Real implementation would call api.post('/chat', { query: input })

            // Mocking delay
            setTimeout(() => {
                const aiMsg = {
                    role: 'assistant',
                    content: `I noticed you asked: "${userMsg.content}". (To make me real, please configure the Gemini API key in the backend!)`
                };
                setMessages(prev => [...prev, aiMsg]);
                setLoading(false);
            }, 1000);

        } catch (err) {
            console.error("Chat Error", err);
            setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I'm having trouble connecting right now." }]);
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-2rem)] max-w-5xl mx-auto">
            <div className="mb-6">
                <h1 className="text-3xl font-bold text-white tracking-tight">AI Advisor</h1>
                <p className="text-secondary mt-1">Ask questions about your spending, budget, or financial advice.</p>
            </div>

            <div className="flex-1 bg-card rounded-2xl border border-gray-800/60 shadow-xl overflow-hidden flex flex-col relative">
                {/* Background Pattern */}
                <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:20px_20px] pointer-events-none"></div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.role === 'assistant' && (
                                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary border border-primary/30 mt-1">
                                    <Bot size={16} />
                                </div>
                            )}

                            <div className={`max-w-[75%] p-4 rounded-2xl shadow-sm ${msg.role === 'user'
                                    ? 'bg-primary text-white rounded-tr-sm'
                                    : 'bg-gray-800/50 border border-gray-700/50 text-gray-200 rounded-tl-sm'
                                }`}>
                                <p className="leading-relaxed text-sm lg:text-base">{msg.content}</p>
                            </div>

                            {msg.role === 'user' && (
                                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-gray-400 mt-1">
                                    <User size={16} />
                                </div>
                            )}
                        </div>
                    ))}
                    {loading && (
                        <div className="flex gap-4 justify-start">
                            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary border border-primary/30 mt-1">
                                <Bot size={16} />
                            </div>
                            <div className="bg-gray-800/50 border border-gray-700/50 px-4 py-3 rounded-2xl rounded-tl-sm flex gap-1.5 items-center h-fit">
                                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"></span>
                                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce delay-100"></span>
                                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce delay-200"></span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 border-t border-gray-800 bg-card/80 backdrop-blur-md">
                    <form onSubmit={handleSend} className="relative max-w-3xl mx-auto flex items-center gap-2">
                        <div className="relative flex-1">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask about your budget..."
                                className="w-full bg-gray-900 border border-gray-700 rounded-xl pl-4 pr-12 py-3.5 text-white placeholder-gray-500 focus:border-primary focus:ring-1 focus:ring-primary focus:outline-none transition-all shadow-inner"
                            />
                            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 text-gray-500 text-xs">
                                <span className="border border-gray-700 rounded px-1.5 py-0.5">Enter</span>
                            </div>
                        </div>
                        <button
                            type="submit"
                            disabled={loading || !input.trim()}
                            className="bg-primary hover:bg-blue-600 text-white p-3.5 rounded-xl transition-all disabled:opacity-50 disabled:hover:bg-primary shadow-lg shadow-blue-500/10"
                        >
                            <Send size={18} />
                        </button>
                    </form>
                    <p className="text-center text-gray-600 text-xs mt-3">
                        AI can make mistakes. Please verify financial details.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Chatbot;
