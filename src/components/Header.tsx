import React, { useState, useEffect } from 'react';
import './Header.css';

const NAV = [
  { id: 'projects', label: 'Work' },
  { id: 'skills', label: 'Stack' },
  { id: 'career', label: 'Career' },
  { id: 'contact', label: 'Contact' },
];

/** Sections with dark / blueish backgrounds — light transparent pill. */
const DARK_SECTION_IDS = ['hero', 'skills', 'contact'];

const Header: React.FC = () => {
  const [onDark, setOnDark] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const update = () => {
      const probeY = 48;
      const overDark = DARK_SECTION_IDS.some((id) => {
        const el = document.getElementById(id);
        if (!el) return false;
        const { top, bottom } = el.getBoundingClientRect();
        return top <= probeY && bottom >= probeY;
      });
      setOnDark(overDark);
    };

    update();
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('scroll', update);
      window.removeEventListener('resize', update);
    };
  }, []);

  const go = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    setMenuOpen(false);
  };

  return (
    <header className={`site-header ${onDark ? 'on-hero' : ''}`}>
      <div className="site-header-pill">
        <button type="button" className="site-brand" onClick={() => go('hero')}>
          SD
        </button>

        <nav className={`site-nav ${menuOpen ? 'open' : ''}`}>
          {NAV.map((item) => (
            <button key={item.id} type="button" onClick={() => go(item.id)}>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="site-header-end">
          <a href="/resume.pdf" className="site-resume" target="_blank" rel="noopener noreferrer">
            Resume
          </a>
          <button
            type="button"
            className={`site-menu-btn ${menuOpen ? 'open' : ''}`}
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Menu"
          >
            <span /><span /><span />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
