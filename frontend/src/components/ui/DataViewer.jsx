import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Check } from 'lucide-react';

const DataViewer = ({ file }) => {
  const [viewMode, setViewMode] = useState('ui'); // 'ui', 'json', 'csv', 'llm'
  const [activeFields, setActiveFields] = useState({});
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (file && file.extractedData) {
      const initialFields = {};
      Object.keys(file.extractedData).forEach(k => initialFields[k] = true);
      setActiveFields(initialFields);
    }
  }, [file]);

  const toggleField = (key) => {
    setActiveFields(prev => ({ ...prev, [key]: !prev[key] }));
  };

  if (!file) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-[#313851]/40 h-full text-center p-8">
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" className="w-16 h-16 mb-4 opacity-50" strokeWidth={1}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m3.75 9v6m3-3H9m1.5-12H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
        </svg>
        <p className="text-lg font-bold text-[#313851]">No document selected</p>
        <p className="text-sm mt-1">Select a processed document from the list to view its extracted data.</p>
      </div>
    );
  }

  if (file.status !== 'done' || !file.extractedData) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-[#313851]/40 h-full text-center p-8">
        <div className="w-12 h-12 border-4 border-[#C2CBD3]/30 border-t-[#6366F1] rounded-full animate-spin mb-4"></div>
        <p className="text-lg font-bold text-[#313851]">Processing...</p>
        <p className="text-sm mt-1">AI is extracting structured data from {file.file.name}</p>
      </div>
    );
  }

  const { extractedData } = file;

  // Create filtered data
  const filteredData = {};
  Object.keys(extractedData).forEach(k => {
    if (activeFields[k]) {
      filteredData[k] = extractedData[k];
    }
  });

  const getFormatOutput = () => {
    if (Object.keys(filteredData).length === 0) return 'No fields selected.';
    
    if (viewMode === 'json') {
      return JSON.stringify(filteredData, null, 2);
    }
    if (viewMode === 'csv') {
      const keys = Object.keys(filteredData);
      const values = Object.values(filteredData).map(v => `"${v}"`);
      return `${keys.join(',')}\n${values.join(',')}`;
    }
    return '';
  };

  const getLLMRawText = () => {
    let type = filteredData.type ? (filteredData.type.charAt(0).toUpperCase() + filteredData.type.slice(1)) : 'Document';
    let msg = type;
    if (filteredData.vendor) msg += ` from ${filteredData.vendor}`;
    if (filteredData.amount) msg += `, total ${filteredData.amount}`;
    if (filteredData.date) msg += `, dated ${filteredData.date}`;
    
    const extraKeys = Object.keys(filteredData).filter(k => !['type', 'vendor', 'amount', 'date'].includes(k));
    if (extraKeys.length > 0) {
      msg += '. Additional details: ' + extraKeys.map(k => `${k} is ${filteredData[k]}`).join(', ');
    }
    return msg + '.';
  };

  const copyToClipboard = () => {
    const text = viewMode === 'llm' ? getLLMRawText() : getFormatOutput();
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderLLMText = () => {
    if (Object.keys(filteredData).length === 0) return <span>No fields selected.</span>;

    const parts = [];
    const type = filteredData.type ? (
      <span key="type" className="bg-[#6366F1]/20 text-[#6366F1] px-1.5 py-0.5 rounded-md font-bold">{filteredData.type.charAt(0).toUpperCase() + filteredData.type.slice(1)}</span>
    ) : 'Document';
    
    parts.push(type);
    
    if (filteredData.vendor) {
      parts.push(" from ");
      parts.push(<span key="vendor" className="bg-[#6366F1]/20 text-[#6366F1] px-1.5 py-0.5 rounded-md font-bold">{filteredData.vendor}</span>);
    }
    
    if (filteredData.amount) {
      parts.push(", total ");
      parts.push(<span key="amount" className="bg-[#6366F1]/20 text-[#6366F1] px-1.5 py-0.5 rounded-md font-bold">{filteredData.amount}</span>);
    }
    
    if (filteredData.date) {
      parts.push(", dated ");
      parts.push(<span key="date" className="bg-[#6366F1]/20 text-[#6366F1] px-1.5 py-0.5 rounded-md font-bold">{filteredData.date}</span>);
    }
    
    const extraKeys = Object.keys(filteredData).filter(k => !['type', 'vendor', 'amount', 'date'].includes(k));
    if (extraKeys.length > 0) {
      parts.push(". Additional details: ");
      extraKeys.forEach((k, idx) => {
        parts.push(`${k} is `);
        parts.push(<span key={k} className="bg-[#C2CBD3]/40 text-[#313851] px-1.5 py-0.5 rounded-md font-bold">{filteredData[k]}</span>);
        if (idx < extraKeys.length - 1) parts.push(", ");
      });
    }
    
    parts.push(".");
    return <span className="text-[15px] leading-8">{parts}</span>;
  };

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex flex-col xl:flex-row xl:items-center justify-between mb-6 pb-4 border-b border-[#C2CBD3]/20 gap-4">
        <div className="min-w-0 pr-4">
          <h3 className="text-xl font-bold text-[#313851] truncate">Extracted Data</h3>
          <p className="text-sm text-[#313851]/60 truncate max-w-xs">{file.file.name}</p>
        </div>
        
        {/* Toggle Formats */}
        <div className="flex bg-[#F6F3ED] rounded-lg p-1 flex-shrink-0 overflow-x-auto">
          {['ui', 'json', 'csv', 'llm'].map(mode => (
            <button 
              key={mode}
              onClick={() => setViewMode(mode)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all capitalize whitespace-nowrap ${viewMode === mode ? 'bg-white text-[#6366F1] shadow-[0_2px_8px_-2px_rgba(49,56,81,0.1)]' : 'text-[#313851]/60 hover:text-[#313851]'}`}
            >
              {mode === 'ui' ? 'Clean UI' : mode === 'llm' ? 'LLM Ready' : mode.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Content Area - Split */}
      <div className="flex flex-1 gap-6 min-h-0 overflow-hidden">
        {/* Left Sidebar - Fields */}
        <div className="w-1/3 flex flex-col border-r border-[#C2CBD3]/20 pr-4 overflow-y-auto pb-4">
          <h4 className="text-[11px] font-bold text-[#313851]/50 uppercase tracking-wider mb-4">Include Fields</h4>
          <div className="flex flex-col gap-4">
            {Object.keys(extractedData).map(key => (
              <label key={key} className="flex items-center gap-3 cursor-pointer group">
                {/* Custom Toggle Switch */}
                <div className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${activeFields[key] ? 'bg-[#6366F1]' : 'bg-[#C2CBD3]'}`}>
                  <input type="checkbox" className="sr-only" checked={!!activeFields[key]} onChange={() => toggleField(key)} />
                  <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${activeFields[key] ? 'translate-x-4' : 'translate-x-0'}`} />
                </div>
                <span className={`text-sm font-medium truncate ${activeFields[key] ? 'text-[#313851]' : 'text-[#313851]/50 line-through'}`}>
                  {key}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Right Content - Viewer */}
        <div className="w-2/3 flex flex-col overflow-y-auto pb-4 pr-2 relative">
          
          {viewMode !== 'ui' && (
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={copyToClipboard}
              className={`absolute top-2 right-2 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
                copied ? 'bg-green-100 text-green-700' : 'bg-[#C2CBD3]/30 text-[#313851]/70 hover:bg-[#C2CBD3]/50 hover:text-[#313851]'
              }`}
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
              {copied ? 'Copied' : 'Copy'}
            </motion.button>
          )}

          {viewMode === 'ui' ? (
            <div className="flex flex-col gap-3">
              <AnimatePresence mode="popLayout">
                {Object.entries(filteredData).map(([key, value]) => (
                  <motion.div 
                    layout
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95, height: 0, padding: 0, marginBottom: 0, border: 0 }}
                    transition={{ duration: 0.2 }}
                    key={key} 
                    className="flex flex-col bg-[#F6F3ED]/40 p-3.5 rounded-xl border border-[#C2CBD3]/20 overflow-hidden"
                  >
                    <span className="text-[10px] font-bold text-[#313851]/50 uppercase tracking-wider mb-1">{key}</span>
                    <span className="text-sm font-semibold text-[#313851] truncate">{value?.toString()}</span>
                  </motion.div>
                ))}
              </AnimatePresence>
              {Object.keys(filteredData).length === 0 && (
                <p className="text-sm text-[#313851]/50 italic mt-2">All fields are hidden.</p>
              )}
            </div>
          ) : viewMode === 'llm' ? (
            <motion.div 
              key="llm"
              initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }}
              className="bg-[#F6F3ED]/40 rounded-xl p-6 text-[#313851] overflow-x-hidden min-h-full border border-[#C2CBD3]/20 pt-12 relative"
            >
              {renderLLMText()}
            </motion.div>
          ) : (
            <motion.div 
              key={viewMode}
              initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }}
              className="bg-[#313851]/5 rounded-xl p-6 font-mono text-[13px] text-[#313851] overflow-x-auto min-h-full pt-12 relative"
            >
              <pre className="whitespace-pre-wrap">{getFormatOutput()}</pre>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DataViewer;
