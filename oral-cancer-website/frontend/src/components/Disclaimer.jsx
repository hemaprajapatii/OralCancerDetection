import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function Disclaimer() {
  return (
    <aside className="disclaimer-card" aria-label="Medical research prototype disclaimer">
      <div className="disclaimer-header">
        <AlertTriangle size={20} />
        <span>Medical Disclaimer</span>
      </div>
      <div className="disclaimer-body">
        This website is an experimental Oral Cancer Detection model. This model is not a certified medical diagnostic system and must not be used to diagnose, exclude, or treat oral cancer. A prediction of NON CANCER does not rule out disease. Anyone with a suspicious oral lesion or persistent symptoms should consult a qualified healthcare professional.
      </div>
    </aside>
  );
}
