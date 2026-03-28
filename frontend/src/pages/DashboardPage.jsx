import React, { useState } from 'react';
import { motion } from 'framer-motion';
import FileUpload from '../components/ui/FileUpload';
import DataViewer from '../components/ui/DataViewer';

const DashboardPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);

  return (
    <div className="min-h-screen bg-[#F6F3ED] text-[#313851] font-sans selection:bg-[#6366F1]/20 flex flex-col">
      {/* Top Navbar */}
      <nav className="sticky top-0 z-50 bg-[#F6F3ED]/80 backdrop-blur-md border-b border-[#C2CBD3]/30">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          {/* Logo Area */}
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="Parseon Logo" className="w-10 h-10 object-contain drop-shadow-sm" />
            <span className="font-bold text-xl tracking-tight">Parseon</span>
          </div>

          {/* Right Navigation */}
          <div className="flex items-center gap-8">
            <a href="#" className="text-sm font-semibold text-[#313851] hover:text-[#6366F1] transition-colors">Dashboard</a>
            <a href="#" className="text-sm font-medium text-[#313851]/70 hover:text-[#6366F1] transition-colors">Docs</a>
            <a href="#" className="text-sm font-medium text-[#313851]/70 hover:text-[#6366F1] transition-colors">Profile</a>
          </div>
        </div>
      </nav>

      {/* Main Area */}
      <motion.main 
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.25, 0.1, 0.25, 1.0] }}
        className="w-full max-w-7xl mx-auto px-6 py-8 flex gap-6 flex-1 h-[calc(100vh-4rem)]"
      >
        
        {/* LEFT PANEL */}
        <div className="w-1/2 flex flex-col bg-white rounded-2xl shadow-[0_4px_20px_-2px_rgba(49,56,81,0.08)] p-8 border border-[#C2CBD3]/20 overflow-y-auto">
          <h2 className="text-2xl font-extrabold text-[#313851] mb-2">Ingest documents</h2>
          <p className="text-[#313851]/70 mb-4">Upload files to extract structured data</p>
          <FileUpload onSelectFile={setSelectedFile} selectedFileId={selectedFile?.id} />
        </div>

        {/* RIGHT PANEL */}
        <div className="w-1/2 flex flex-col bg-white rounded-2xl shadow-[0_4px_20px_-2px_rgba(49,56,81,0.08)] p-8 border border-[#C2CBD3]/20 overflow-y-auto">
           <DataViewer file={selectedFile} />
        </div>

      </motion.main>
    </div>
  );
};

export default DashboardPage;
