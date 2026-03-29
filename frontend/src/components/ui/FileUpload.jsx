import React, { useState, useRef, useCallback } from 'react';
import { UploadCloud, File, X, CheckCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

const FileUpload = ({ onSelectFile, selectedFileId }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState([]);
  const [filter, setFilter] = useState('all'); // 'all', 'processing', 'completed'
  const fileInputRef = useRef(null);

  const formatSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  // Send files to backend for OCR
  const processFiles = useCallback(async (newFiles) => {
    const validFiles = Array.from(newFiles).filter(file => {
      return (
        file.type.startsWith('image/') ||
        file.type === 'application/pdf' ||
        file.type.includes('word') ||
        file.type.includes('document')
      );
    });
    if (validFiles.length === 0) return;

    // Add files to UI as uploading
    const newFileObjects = validFiles.map(file => ({
      id: Math.random().toString(36).substring(7),
      file,
      status: 'uploading',
      progress: 0
    }));
    setFiles(prev => [...prev, ...newFileObjects]);

    // Prepare FormData for multi-upload
    const formData = new FormData();
    validFiles.forEach(f => formData.append('files', f));

    // Update status to processing
    setFiles(prev => prev.map(f => newFileObjects.find(nf => nf.id === f.id) ? { ...f, status: 'processing', progress: 100 } : f));

    try {
      const response = await fetch('/predict-multi', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (data && Array.isArray(data.results)) {
        setFiles(prev => prev.map(f => {
          const result = data.results.find(r => r.filename === f.file.name);
          if (!result) return f;
          if (result.error) {
            return { ...f, status: 'failed', error: result.error };
          }
          return {
            ...f,
            status: 'done',
            extractedData: result,
            classification: { type: 'OCR', classes: 'bg-blue-100 text-blue-700 border-blue-200' }
          };
        }));
      } else {
        setFiles(prev => prev.map(f => ({ ...f, status: 'failed', error: 'No OCR result' })));
      }
    } catch (err) {
      setFiles(prev => prev.map(f => ({ ...f, status: 'failed', error: 'Server error' })));
    }
  }, []);

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const onFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
    // reset input so the same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (idToRemove) => {
    setFiles(prev => prev.filter(f => f.id !== idToRemove));
  };

  return (
    <div className="w-full mt-8">
      {/* Upload Zone */}
      <div 
        className={`w-full border-2 border-dashed rounded-2xl p-10 md:p-14 flex flex-col items-center justify-center transition-all duration-300 cursor-pointer group
          ${isDragging ? 'border-[#6366F1] bg-[#F6F3ED]' : 'border-[#C2CBD3] hover:border-[#6366F1] hover:bg-[#F6F3ED]/50'}
        `}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="w-16 h-16 bg-white rounded-2xl flex items-center justify-center mb-4 shadow-sm group-hover:scale-105 transition-transform duration-300">
          <UploadCloud size={32} className="text-[#6366F1]" />
        </div>
        <p className="text-lg font-bold text-[#313851] mb-1">
          Drag & drop files or click to upload
        </p>
        <p className="text-sm font-medium text-[#313851]/60">
          Supports Images, PDFs, and Docs
        </p>

        <input 
          type="file" 
          ref={fileInputRef}
          className="hidden"
          multiple
          accept="image/*,.pdf,.doc,.docx"
          onChange={onFileSelect}
        />
      </div>

      {/* Dashboard Stats & Filters */}
      <div className="mt-8 mb-4 flex flex-col gap-6">
        <div className="grid grid-cols-3 gap-4">
          <motion.div whileHover={{ y: -2, boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.05)' }} className="bg-[#F6F3ED]/60 rounded-xl p-4 border border-[#C2CBD3]/30 flex flex-col transition-all duration-300">
            <span className="text-xs font-bold text-[#313851]/50 uppercase tracking-wider mb-1">Completed</span>
            <span className="text-2xl font-extrabold text-[#6366F1]">{files.filter(f => f.status === 'done').length}</span>
          </motion.div>
          <motion.div whileHover={{ y: -2, boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.05)' }} className="bg-[#F6F3ED]/60 rounded-xl p-4 border border-[#C2CBD3]/30 flex flex-col transition-all duration-300">
            <span className="text-xs font-bold text-[#313851]/50 uppercase tracking-wider mb-1">Processing</span>
            <span className="text-2xl font-extrabold text-orange-400">{files.filter(f => f.status === 'processing' || f.status === 'uploading').length}</span>
          </motion.div>
          <motion.div whileHover={{ y: -2, boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.05)' }} className="bg-[#F6F3ED]/60 rounded-xl p-4 border border-[#C2CBD3]/30 flex flex-col transition-all duration-300">
            <span className="text-xs font-bold text-[#313851]/50 uppercase tracking-wider mb-1">Failed</span>
            <span className="text-2xl font-extrabold text-red-500">{files.filter(f => f.status === 'failed').length}</span>
          </motion.div>
        </div>

        <div className="flex items-center justify-between border-b border-[#C2CBD3]/20 pb-2">
          <h3 className="text-lg font-bold text-[#313851]">File History</h3>
          <div className="flex bg-[#F6F3ED] rounded-lg p-1">
            {['all', 'processing', 'completed'].map(f => (
              <button 
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-all capitalize ${filter === f ? 'bg-white text-[#6366F1] shadow-sm' : 'text-[#313851]/60 hover:text-[#313851]'}`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* File List */}
      <div className="flex flex-col gap-3 min-h-[200px]">
        {files.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} 
            className="flex flex-col items-center justify-center p-8 text-center text-[#313851]/40 border-2 border-dashed border-[#C2CBD3]/30 rounded-xl bg-[#F6F3ED]/20"
          >
            <p className="text-sm font-medium">No files uploaded yet.</p>
          </motion.div>
        ) : (
          files.filter(f => {
            if (filter === 'processing') return f.status === 'processing' || f.status === 'uploading';
            if (filter === 'completed') return f.status === 'done';
            return true;
          }).map((fileObj) => (
            <motion.div 
              layout
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              whileHover={(fileObj.status === 'done' || fileObj.status === 'failed') ? { scale: 1.01 } : {}}
              whileTap={(fileObj.status === 'done' || fileObj.status === 'failed') ? { scale: 0.99 } : {}}
              key={fileObj.id} 
              onClick={() => {
                if(fileObj.status === 'done' || fileObj.status === 'failed') {
                  onSelectFile && onSelectFile(fileObj);
                }
              }}
              className={`relative overflow-hidden bg-white border rounded-xl p-4 transition-colors ${
                (fileObj.status === 'done' || fileObj.status === 'failed') ? 'cursor-pointer hover:border-[#6366F1]/50' : 'opacity-90'
              } ${
                selectedFileId === fileObj.id 
                  ? 'border-[#6366F1] shadow-[0_4px_14px_-2px_rgba(99,102,241,0.2)]' 
                  : 'border-[#C2CBD3]/30 shadow-[0_2px_10px_-2px_rgba(49,56,81,0.05)]'
              }`}
            >
              
              {/* Progress Bar Background */}
              {fileObj.status === 'uploading' && (
                <div 
                  className="absolute bottom-0 left-0 h-1 bg-[#6366F1] transition-all duration-300 ease-out"
                  style={{ width: `${fileObj.progress}%` }}
                />
              )}
              
              <div className="flex items-start justify-between relative z-10 w-full">
                <div className="flex items-start gap-4 flex-1 overflow-hidden">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 ${
                    fileObj.status === 'done' ? 'bg-green-50 text-green-500' : 
                    fileObj.status === 'failed' ? 'bg-red-50 text-red-500' : 'bg-[#F6F3ED] text-[#6366F1]'
                  }`}>
                    {fileObj.status === 'done' ? (
                      <CheckCircle size={20} />
                    ) : fileObj.status === 'failed' ? (
                      <X size={20} />
                    ) : (
                      <File size={20} />
                    )}
                  </div>
                  
                  <div className="flex flex-col text-left flex-1 min-w-0">
                    <div className="flex items-center gap-3 w-full">
                      <span className={`text-sm font-semibold truncate max-w-[150px] sm:max-w-[200px] ${fileObj.status === 'failed' ? 'text-red-500' : 'text-[#313851]'}`}>{fileObj.file.name}</span>
                      {fileObj.status === 'done' && fileObj.classification && (
                        <motion.span 
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ type: "spring", stiffness: 300, damping: 20 }}
                          className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded border ${fileObj.classification.classes} flex-shrink-0`}
                        >
                          {fileObj.classification.type}
                        </motion.span>
                      )}
                    </div>
                    
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs font-medium text-[#313851]/60">{formatSize(fileObj.file.size)}</span>
                      
                      {fileObj.status === 'uploading' && (
                        <>
                          <span className="w-1 h-1 bg-[#C2CBD3] rounded-full"></span>
                          <span className="text-xs font-medium text-[#6366F1]">{fileObj.progress}%</span>
                        </>
                      )}
                      
                      {fileObj.status === 'processing' && (
                        <>
                          <span className="w-1 h-1 bg-[#C2CBD3] rounded-full"></span>
                          <span className="text-xs font-medium text-[#6366F1] flex items-center gap-1.5 animate-pulse">
                            <Loader2 size={12} className="animate-spin" /> Processing with AI...
                          </span>
                        </>
                      )}
                      
                      {fileObj.status === 'done' && (
                        <>
                          <span className="w-1 h-1 bg-[#C2CBD3] rounded-full"></span>
                          <span className="text-xs font-medium text-green-500">Ready</span>
                        </>
                      )}

                      {fileObj.status === 'failed' && (
                        <>
                          <span className="w-1 h-1 bg-[#C2CBD3] rounded-full"></span>
                          <span className="text-xs font-medium text-red-500">Extraction failed</span>
                        </>
                      )}
                    </div>

                    {/* Skeleton Loader inside the file item while processing */}
                    {fileObj.status === 'processing' && (
                      <div className="w-full max-w-[200px] mt-3 space-y-2">
                        <div className="h-1.5 w-full bg-[#F6F3ED] rounded-full overflow-hidden">
                          <div className="h-full bg-[#E5E7EB] w-2/3 animate-pulse rounded-full"></div>
                        </div>
                        <div className="h-1.5 w-4/5 bg-[#F6F3ED] rounded-full overflow-hidden">
                          <div className="h-full bg-[#E5E7EB] w-1/2 animate-pulse rounded-full" style={{ animationDelay: '150ms' }}></div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <button 
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(fileObj.id);
                  }}
                  className="w-8 h-8 rounded-full flex items-center justify-center text-[#313851]/40 hover:text-[#313851] hover:bg-[#F6F3ED] transition-colors ml-4 flex-shrink-0"
                  title="Remove file"
                >
                  <X size={18} />
                </button>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
};

export default FileUpload;
