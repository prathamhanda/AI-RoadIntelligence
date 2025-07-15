import React, { useState } from 'react';
import { Upload, Camera, Monitor, BarChart3, Settings, FileImage, Link, Play, Trash2, Eye, EyeOff } from 'lucide-react';

interface PerformanceMetrics {
  mAP50: number;
  precision: number;
  recall: number;
}

interface ThresholdControls {
  confidence: number;
  overlap: number;
}

interface UploadedFile {
  id: string;
  name: string;
  url: string;
  type: 'image' | 'video';
  size: number;
  status: 'uploading' | 'completed' | 'error';
  progress: number;
}

function App() {
  const [showGraph, setShowGraph] = useState(false);
  const [thresholds, setThresholds] = useState<ThresholdControls>({
    confidence: 50,
    overlap: 50,
  });
  const [labelMode, setLabelMode] = useState('Draw Confidence');
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [urlInput, setUrlInput] = useState('');

  const performanceMetrics: PerformanceMetrics = {
    mAP50: 67.7,
    precision: 77.9,
    recall: 57.1,
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files);
    handleFileUpload(files);
  };

  const handleFileUpload = (files: File[]) => {
    files.forEach((file) => {
      const newFile: UploadedFile = {
        id: Date.now() + Math.random().toString(),
        name: file.name,
        url: URL.createObjectURL(file),
        type: file.type.startsWith('image/') ? 'image' : 'video',
        size: file.size,
        status: 'uploading',
        progress: 0,
      };
      
      setUploadedFiles(prev => [...prev, newFile]);
      
      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadedFiles(prev => prev.map(f => {
          if (f.id === newFile.id) {
            const newProgress = f.progress + Math.random() * 30;
            if (newProgress >= 100) {
              clearInterval(interval);
              return { ...f, progress: 100, status: 'completed' };
            }
            return { ...f, progress: newProgress };
          }
          return f;
        }));
      }, 200);
    });
  };

  const handleUrlSubmit = () => {
    if (urlInput.trim()) {
      const newFile: UploadedFile = {
        id: Date.now() + Math.random().toString(),
        name: 'URL Upload',
        url: urlInput,
        type: 'image',
        size: 0,
        status: 'completed',
        progress: 100,
      };
      setUploadedFiles(prev => [...prev, newFile]);
      setUrlInput('');
    }
  };

  const removeFile = (id: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== id));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return 'N/A';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getProgressColor = (progress: number) => {
    if (progress < 30) return 'from-cyan-500 to-blue-500';
    if (progress < 70) return 'from-purple-500 to-pink-500';
    return 'from-green-500 to-emerald-500';
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                <Camera className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Smart Traffic Congestion System</h1>
                <p className="text-sm text-gray-400">Model Trained on traffic-detection dataset</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-300">Trained with 12,000+ images</span>
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-purple-500 to-pink-500 text-white">
                YOLOv8m Model Upload
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Performance Metrics */}
            <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 shadow-xl">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white">Performance Metrics</h2>
                <button
                  onClick={() => setShowGraph(!showGraph)}
                  className="inline-flex items-center px-4 py-2 border border-gray-600 text-sm font-medium rounded-lg text-gray-300 bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-cyan-500 transition-all duration-200"
                >
                  {showGraph ? <EyeOff className="h-4 w-4 mr-2" /> : <Eye className="h-4 w-4 mr-2" />}
                  {showGraph ? 'Hide Graph' : 'View Graph'}
                </button>
              </div>
              
              <div className="grid grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
                    {performanceMetrics.mAP50}%
                  </div>
                  <div className="text-sm text-gray-400 mb-3">mAP@50</div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-1000 ease-out"
                      style={{ width: `${performanceMetrics.mAP50}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent mb-2">
                    {performanceMetrics.precision}%
                  </div>
                  <div className="text-sm text-gray-400 mb-3">Precision</div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full transition-all duration-1000 ease-out"
                      style={{ width: `${performanceMetrics.precision}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent mb-2">
                    {performanceMetrics.recall}%
                  </div>
                  <div className="text-sm text-gray-400 mb-3">Recall</div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-1000 ease-out"
                      style={{ width: `${performanceMetrics.recall}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {showGraph && (
                <div className="mt-6 p-6 bg-gray-900 rounded-lg border border-gray-600">
                  <div className="text-center text-gray-400 py-12">
                    <BarChart3 className="h-16 w-16 mx-auto mb-4 text-gray-500" />
                    <p className="text-lg">Performance graph visualization would appear here</p>
                    <p className="text-sm mt-2">Interactive charts showing model performance over time</p>
                  </div>
                </div>
              )}
            </div>

            {/* Inference Area */}
            <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 shadow-xl">
              <h2 className="text-xl font-semibold text-white mb-2">Model Inference</h2>
              <p className="text-sm text-gray-400 mb-6">Add test images to preview model inference</p>
              
              {/* File Upload Area */}
              <div
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                  dragActive 
                    ? 'border-cyan-500 bg-cyan-500/10 shadow-lg shadow-cyan-500/20' 
                    : 'border-gray-600 hover:border-gray-500 bg-gray-900/50'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center">
                  <Upload className="h-8 w-8 text-white" />
                </div>
                <p className="text-lg font-medium text-white mb-2">
                  Drop your file here
                </p>
                <p className="text-sm text-gray-400 mb-6">
                  Supports JPG, PNG, MP4, MOV files
                </p>
                
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <label className="inline-flex items-center px-6 py-3 border border-gray-600 text-sm font-medium rounded-lg text-gray-300 bg-gray-700 hover:bg-gray-600 cursor-pointer transition-all duration-200 hover:scale-105">
                    <FileImage className="h-4 w-4 mr-2" />
                    Choose File
                    <input
                      type="file"
                      className="hidden"
                      multiple
                      accept="image/*,video/*"
                      onChange={(e) => e.target.files && handleFileUpload(Array.from(e.target.files))}
                    />
                  </label>
                  
                  <button className="inline-flex items-center px-6 py-3 border border-gray-600 text-sm font-medium rounded-lg text-gray-300 bg-gray-700 hover:bg-gray-600 transition-all duration-200 hover:scale-105">
                    <Camera className="h-4 w-4 mr-2" />
                    Try with Webcam
                  </button>
                  
                  <button className="inline-flex items-center px-6 py-3 border border-gray-600 text-sm font-medium rounded-lg text-gray-300 bg-gray-700 hover:bg-gray-600 transition-all duration-200 hover:scale-105">
                    <Monitor className="h-4 w-4 mr-2" />
                    Try on Your Machine
                  </button>
                </div>
              </div>

              {/* URL Input */}
              <div className="mt-6">
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Or paste an image URL / YouTube link
                </label>
                <div className="flex gap-3">
                  <input
                    type="url"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="https://example.com/image.jpg"
                    className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-200"
                  />
                  <button
                    onClick={handleUrlSubmit}
                    className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-medium rounded-lg hover:from-cyan-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-cyan-500 transition-all duration-200 hover:scale-105"
                  >
                    <Link className="h-4 w-4 mr-2" />
                    Add URL
                  </button>
                </div>
              </div>

              {/* Uploaded Files */}
              {uploadedFiles.length > 0 && (
                <div className="mt-8">
                  <h3 className="text-sm font-medium text-gray-300 mb-4">Uploaded Files</h3>
                  <div className="space-y-3">
                    {uploadedFiles.map((file) => (
                      <div key={file.id} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                              {file.type === 'image' ? (
                                <FileImage className="h-5 w-5 text-white" />
                              ) : (
                                <Play className="h-5 w-5 text-white" />
                              )}
                            </div>
                            <div>
                              <p className="text-sm font-medium text-white">{file.name}</p>
                              <p className="text-xs text-gray-400">{formatFileSize(file.size)}</p>
                            </div>
                          </div>
                          <button
                            onClick={() => removeFile(file.id)}
                            className="text-red-400 hover:text-red-300 p-2 rounded-lg hover:bg-red-500/10 transition-all duration-200"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                        
                        {file.status === 'uploading' && (
                          <div className="w-full bg-gray-600 rounded-full h-2">
                            <div 
                              className={`bg-gradient-to-r ${getProgressColor(file.progress)} h-2 rounded-full transition-all duration-300`}
                              style={{ width: `${file.progress}%` }}
                            ></div>
                          </div>
                        )}
                        
                        {file.status === 'completed' && (
                          <div className="flex items-center text-xs text-green-400">
                            <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                            Upload completed
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Sidebar Controls */}
          <div className="space-y-6">
            {/* Threshold Controls */}
            <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 shadow-xl">
              <div className="flex items-center mb-6">
                <Settings className="h-5 w-5 text-cyan-400 mr-3" />
                <h3 className="text-lg font-semibold text-white">Threshold Controls</h3>
              </div>
              
              <div className="space-y-8">
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <label className="text-sm font-medium text-gray-300">
                      Confidence Threshold
                    </label>
                    <span className="text-sm font-bold text-cyan-400">{thresholds.confidence}%</span>
                  </div>
                  <div className="relative">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={thresholds.confidence}
                      onChange={(e) => setThresholds(prev => ({ ...prev, confidence: parseInt(e.target.value) }))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb-cyan"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-2">
                      <span>0%</span>
                      <span>100%</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <label className="text-sm font-medium text-gray-300">
                      Overlap Threshold
                    </label>
                    <span className="text-sm font-bold text-purple-400">{thresholds.overlap}%</span>
                  </div>
                  <div className="relative">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={thresholds.overlap}
                      onChange={(e) => setThresholds(prev => ({ ...prev, overlap: parseInt(e.target.value) }))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb-purple"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-2">
                      <span>0%</span>
                      <span>100%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Label Settings */}
            <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 shadow-xl">
              <h3 className="text-lg font-semibold text-white mb-4">Label Settings</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Label Display Mode
                </label>
                <select
                  value={labelMode}
                  onChange={(e) => setLabelMode(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="Draw Confidence">Draw Confidence</option>
                  <option value="Bounding Box Only">Bounding Box Only</option>
                  <option value="Hide Labels">Hide Labels</option>
                </select>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 shadow-xl">
              <h3 className="text-lg font-semibold text-white mb-6">Quick Actions</h3>
              
              <div className="space-y-3">
                <button className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 px-4 rounded-lg hover:from-cyan-600 hover:to-blue-700 transition-all duration-200 font-medium hover:scale-105 shadow-lg">
                  Run Inference
                </button>
                <button className="w-full bg-gray-700 text-gray-300 py-3 px-4 rounded-lg hover:bg-gray-600 transition-all duration-200 font-medium border border-gray-600">
                  Reset Settings
                </button>
                <button className="w-full bg-gray-700 text-gray-300 py-3 px-4 rounded-lg hover:bg-gray-600 transition-all duration-200 font-medium border border-gray-600">
                  Export Results
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;