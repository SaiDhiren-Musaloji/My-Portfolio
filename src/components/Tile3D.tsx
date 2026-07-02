import React, { useRef, useCallback } from 'react';
import './Tile3D.css';

interface Tile3DProps {
  children: React.ReactNode;
  className?: string;
  intensity?: number;
  onClick?: () => void;
  role?: string;
  tabIndex?: number;
  onKeyDown?: (e: React.KeyboardEvent) => void;
}

const Tile3D: React.FC<Tile3DProps> = ({
  children,
  className = '',
  intensity = 10,
  onClick,
  role,
  tabIndex,
  onKeyDown,
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const shineRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const el = ref.current;
      const shine = shineRef.current;
      if (!el) return;

      const rect = el.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      const rotateY = (x - 0.5) * intensity * 2;
      const rotateX = (0.5 - y) * intensity * 2;

      el.style.transform = `perspective(900px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(0)`;

      if (shine) {
        shine.style.background = `radial-gradient(circle at ${x * 100}% ${y * 100}%, rgba(255,255,255,0.18) 0%, transparent 60%)`;
        shine.style.opacity = '1';
      }
    },
    [intensity]
  );

  const handleMouseLeave = useCallback(() => {
    const el = ref.current;
    const shine = shineRef.current;
    if (el) el.style.transform = '';
    if (shine) shine.style.opacity = '0';
  }, []);

  return (
    <div
      ref={ref}
      className={`tile-3d ${className}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      role={role}
      tabIndex={tabIndex}
      onKeyDown={onKeyDown}
    >
      <div className="tile-3d-surface">
        <div className="tile-3d-content">{children}</div>
        <div ref={shineRef} className="tile-3d-shine" aria-hidden="true" />
      </div>
      <div className="tile-3d-shadow" aria-hidden="true" />
    </div>
  );
};

export default Tile3D;
