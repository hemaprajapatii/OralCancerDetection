import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, FileImage, Trash2, ArrowRight, Loader2, AlertCircle } from 'lucide-react';

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];

export default function ImageUploader({ onAnalyze, isLoading, isBackendReady }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileDimensions, setFileDimensions] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // Clean up object URL when component unmounts or previewUrl changes
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const validateAndSetFile = (file) => {
    setValidationError(null);

    if (!file) {
      return;
    }

    // Check file size
    if (file.size === 0) {
      setValidationError('The selected file is empty (0 bytes). Please select a valid oral image.');
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      setValidationError(`The selected file is ${sizeMb} MB, which exceeds the maximum allowed limit of 10 MB.`);
      return;
    }

    // Check extension
    const name = file.name.toLowerCase();
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext));
    if (!hasValidExt) {
      setValidationError(`Unsupported format. Please select an image with extension: ${ALLOWED_EXTENSIONS.join(', ')}.`);
      return;
    }

    // Revoke previous URL
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objectUrl);

    // Read natural dimensions
    const img = new Image();
    img.onload = () => {
      setFileDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = objectUrl;
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    validateAndSetFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const file = e.dataTransfer.files?.[0];
    validateAndSetFile(file);
  };

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileDimensions(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyzeClick = () => {
    if (!selectedFile || isLoading) return;
    onAnalyze(selectedFile);
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="card-panel">
      <h2 className="card-title">
        <FileImage size={20} className="text-teal" />
        Oral Image Upload
      </h2>
      <p className="card-subtitle">
        Upload a complete clinical photograph of the oral cavity (.jpg, .jpeg, .png, .webp &le; 10 MB).
      </p>

      {/* Hidden native input */}
      <input
        ref={fileInputRef}
        type="file"
        id="oral-image-input"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        aria-label="Upload oral photograph file"
      />

      {/* Dropzone view if no file selected */}
      {!previewUrl && (
        <div
          className={`dropzone ${isDragOver ? 'active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          tabIndex={0}
          role="button"
          aria-label="Drag and drop an oral image here, or click to browse"
        >
          <UploadCloud className="dropzone-icon" />
          <div className="dropzone-text-primary">Drag & drop oral photograph here</div>
          <div className="dropzone-text-secondary">Supported formats: JPG, JPEG, PNG, WEBP (Max 10 MB)</div>
          <button
            type="button"
            className="browse-btn"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
          >
            Browse from Computer
          </button>
        </div>
      )}

      {/* Preview view when file selected */}
      {previewUrl && (
        <div className="preview-container">
          <div className="preview-image-wrapper">
            <img src={previewUrl} alt="Selected oral cavity photograph preview" className="preview-image" />
          </div>

          <div className="preview-metadata">
            <span className="preview-filename" title={selectedFile?.name}>
              {selectedFile?.name}
            </span>
            <span>
              {fileDimensions ? `${fileDimensions.width} \u00D7 ${fileDimensions.height} px` : ''} &bull;{' '}
              {formatFileSize(selectedFile?.size)}
            </span>
          </div>
        </div>
      )}

      {/* Validation Error Banner */}
      {validationError && (
        <div className="alert-banner alert-danger" role="alert">
          <AlertCircle size={18} className="alert-icon" />
          <div>{validationError}</div>
        </div>
      )}

      {/* Actions */}
      <div className="button-row" style={{ marginTop: previewUrl ? '16px' : '20px' }}>
        <button
          type="button"
          className="btn-primary"
          onClick={handleAnalyzeClick}
          disabled={!selectedFile || isLoading || !isBackendReady}
          title={!isBackendReady ? 'Waiting for backend connection' : 'Analyze image with MobileNetV2'}
        >
          {isLoading ? (
            <>
              <Loader2 size={18} className="spinner" />
              <span>Analyzing Image...</span>
            </>
          ) : (
            <>
              <span>Analyze Image</span>
              <ArrowRight size={18} />
            </>
          )}
        </button>

        {previewUrl && (
          <button
            type="button"
            className="btn-secondary"
            onClick={handleClear}
            disabled={isLoading}
            aria-label="Remove image and reset upload"
          >
            <Trash2 size={16} />
            <span>Reset</span>
          </button>
        )}
      </div>
    </div>
  );
}
