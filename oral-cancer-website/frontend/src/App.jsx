import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import ModelOverview from './components/ModelOverview';
import ImageUploader from './components/ImageUploader';
import ResultDisplay from './components/ResultDisplay';
import Disclaimer from './components/Disclaimer';
import { AlertCircle } from 'lucide-react';

const BACKEND_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'https://oralcancerdetection-2.onrender.com'
).replace(/\/$/, '');

export default function App() {
  const [backendStatus, setBackendStatus] = useState({
    connected: false,
    modelLoaded: false,
    loading: true,
    error: null,
  });

  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [uploaderKey, setUploaderKey] = useState(0);

  // Check backend health
  const checkHealth = useCallback(async (isInitial = false) => {
    if (!isInitial) {
      setBackendStatus((prev) => ({ ...prev, loading: true, error: null }));
    }
    try {
      const response = await fetch(`${BACKEND_BASE_URL}/api/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Backend returned status ${response.status}`);
      }

      const data = await response.json();
      setBackendStatus({
        connected: true,
        modelLoaded: Boolean(data.model_loaded),
        loading: false,
        error: null,
      });
    } catch (err) {
      setBackendStatus({
        connected: false,
        modelLoaded: false,
        loading: false,
        error: err.message || 'Could not reach backend service',
      });
    }
  }, []);

  useEffect(() => {
    checkHealth(true);
  }, [checkHealth]);

  // Handle Image Analysis Submission
  const handleAnalyzeImage = async (file) => {
    setIsAnalyzing(true);
    setApiError(null);

    const formData = new FormData();
    // Use field name 'image' as mandated by backend contract
    formData.append('image', file);

    try {
      // NOTE: Do NOT set Content-Type header manually; browser must set multipart boundary
      const response = await fetch(`${BACKEND_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      let responseData;
      try {
        responseData = await response.json();
      } catch {
        throw new Error('Received an invalid or non-JSON response from server.');
      }

      if (!response.ok || !responseData.success) {
        const errorMsg =
          responseData?.error ||
          (response.status === 413
            ? 'The image exceeds the 10 MB maximum upload limit.'
            : response.status === 503
              ? 'The model is not loaded or the backend is currently unavailable.'
              : `Analysis failed (HTTP ${response.status}).`);
        throw new Error(errorMsg);
      }

      // Valid prediction payload received
      setAnalysisResult(responseData);
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setApiError('Unable to connect to the backend server at http://127.0.0.1:5000. Please ensure the backend is started.');
        setBackendStatus((prev) => ({ ...prev, connected: false, modelLoaded: false }));
      } else {
        setApiError(err.message || 'An unexpected error occurred during image analysis.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setApiError(null);
    setUploaderKey((prev) => prev + 1);
  };

  return (
    <div className="app-container">
      {/* Navigation & Header */}
      <Header backendStatus={backendStatus} onRefreshStatus={() => checkHealth(false)} />

      <main id="main-content">
        {/* Technical Architecture Context */}
        <ModelOverview />

        {/* Global Error Banner */}
        {apiError && (
          <div className="alert-banner alert-danger" style={{ marginBottom: '24px' }} role="alert">
            <AlertCircle size={20} className="alert-icon" />
            <div>
              <strong>Analysis Request Failed:</strong> {apiError}
            </div>
          </div>
        )}

        {/* Workspace: Upload on left, Results on right */}
        <div className="workspace-grid">
          <ImageUploader
            key={uploaderKey}
            onAnalyze={handleAnalyzeImage}
            isLoading={isAnalyzing}
            isBackendReady={backendStatus.connected && backendStatus.modelLoaded}
          />

          <ResultDisplay
            result={analysisResult}
            onReset={handleReset}
          />
        </div>

        {/* Safety & Medical Disclaimer */}
        <Disclaimer />
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Standalone Oral Cancer Detection &bull; Deep Learning MobileNetV2 Architecture &bull;
        </p>
      </footer>
    </div>
  );
}
