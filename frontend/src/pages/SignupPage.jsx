import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Chrome, Apple } from 'lucide-react';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import Logo from '../components/ui/Logo';

const SignupPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="card-container"
    >
      <Logo />
      <div className="text-center mb-8">
        <h1 className="text-2xl font-extrabold text-primary mb-2">Create an account</h1>
        <p className="text-sm text-primary/70">Sign up to get started today.</p>
      </div>

      <form className="space-y-5" onSubmit={(e) => e.preventDefault()}>
        <Input label="Full Name" type="text" placeholder="John Doe" required />
        <Input label="Email" type="email" placeholder="@uxintace.com" required />
        <Input label="Password" type="password" placeholder="••••••••••" required />
        <Input label="Confirm Password" type="password" placeholder="••••••••••" required />

        <Button type="submit" className="mt-4">Register</Button>

        <div className="relative flex items-center gap-4 py-3">
          <div className="flex-1 h-px bg-secondary/50"></div>
          <span className="text-[10px] font-bold text-secondary uppercase tracking-[0.2em]">OR</span>
          <div className="flex-1 h-px bg-secondary/50"></div>
        </div>

        <div className="flex flex-col gap-3">
          <Button variant="social" className="h-11">
            <Apple size={20} className="text-black" />
          </Button>
          <Button variant="social" className="h-11">
            <span className="flex items-center justify-center">
               <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M23.745 12.27c0-.825-.075-1.62-.21-2.385H12.27v4.515h6.435c-.27 1.455-1.08 2.685-2.31 3.51v2.91h3.735c2.19-2.01 3.45-4.995 3.45-8.55z"/>
                  <path fill="#34A853" d="M12.27 24c3.24 0 5.955-1.08 7.935-2.91l-3.735-2.91c-1.08.72-2.46 1.155-4.2 1.155-3.225 0-5.97-2.175-6.945-5.085H1.47v3.015C3.48 21.285 7.56 24 12.27 24z"/>
                  <path fill="#FBBC05" d="M5.325 15.165c-.255-.72-.405-1.5-.405-2.31 0-.81.15-1.59.405-2.31V7.53H1.47C.645 9.15 0 10.965 0 12.855c0 1.89.645 3.705 1.47 5.325l3.855-3.015z"/>
                  <path fill="#EA4335" d="M12.27 4.755c1.755 0 3.345.6 4.59 1.785l3.45-3.45C18.225 1.14 15.51 0 12.27 0 7.56 0 3.48 2.715 1.47 6.705l3.855 3.015c.975-2.91 3.72-5.085 6.945-5.085z"/>
                </svg>
            </span>
          </Button>
        </div>
      </form>

      <p className="mt-8 text-center text-sm text-primary/70">
        Already have an account?{' '}
        <Link to="/login" className="font-semibold text-accent hover:text-accent-dark transition-colors">
          Login
        </Link>
      </p>
    </motion.div>
  );
};

export default SignupPage;
