import React from 'react';
import { Activity, RefreshCw, AlertCircle } from 'lucide-react';

export default function Header({ backendStatus, onRefreshStatus }) {
  return (
    <header className="app-header">
      <div className="brand-wrapper">
        <div className="brand-icon" aria-hidden="true">
          <Activity size={26} strokeWidth={2.2} />
        </div>
        <div className="brand-title-group">
          <h1>Oral Cancer Detection</h1>
          <div className="brand-sub">
            <span className="badge-tag">Prototype Model</span>
            <span>MobileNetV2</span>
          </div>
        </div>
      </div>

      <div className="status-container">
        {backendStatus.loading && (
          <div className="backend-status-pill status-checking" role="status" aria-live="polite">
            <span className="pulse-dot"></span>
            <span>Checking backend...</span>
          </div>
        )}

        {!backendStatus.loading && backendStatus.connected && backendStatus.modelLoaded && (
          <div className="backend-status-pill status-online" role="status" aria-live="polite">
            <span className="pulse-dot"></span>
            <span>Model Ready (127.0.0.1:5000)</span>
          </div>
        )}

        {!backendStatus.loading && backendStatus.connected && !backendStatus.modelLoaded && (
          <div className="backend-status-pill status-checking" role="status" aria-live="polite">
            <AlertCircle size={14} />
            <span>Backend Online (Model Degraded)</span>
            <button
              onClick={onRefreshStatus}
              className="btn-retry-status"
              title="Retry Connection"
              aria-label="Retry connection to backend"
            >
              <RefreshCw size={12} />
            </button>
          </div>
        )}

        {!backendStatus.loading && !backendStatus.connected && (
          <div className="backend-status-pill status-offline" role="status" aria-live="polite">
            <span className="pulse-dot"></span>
            <span>Backend Offline</span>
            <button
              onClick={onRefreshStatus}
              className="btn-retry-status"
              title="Retry Connection"
              aria-label="Retry connection to backend"
            >
              <RefreshCw size={12} />
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
