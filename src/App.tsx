import React from 'react';
import './App.css';
import { Analytics } from '@vercel/analytics/react';
import Header from './components/Header';
import Hero from './sections/Hero';
import About from './sections/About';
import Experience from './sections/Experience';
import Education from './sections/Education';
import Projects from './sections/Projects';
import Skills from './sections/Skills';
import Certificates from './sections/Certificates';
import Contact from './sections/Contact';
import Footer from './components/Footer';

function App() {
  return (
    <div className="App">
      <Header />
      <main>
        <Hero />
        <About />
        <Experience />
        <Education />
        <Projects />
        <Skills />
        <Certificates />
        <Contact />
      </main>
      <Footer />
      <Analytics />
    </div>
  );
}

export default App;
