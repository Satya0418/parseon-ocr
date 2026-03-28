import { motion } from 'framer-motion';

const Button = ({ children, variant = 'primary', className = '', ...props }) => {
  const variants = {
    primary: 'bg-primary text-white hover:bg-gray-800 shadow-sm',
    secondary: 'bg-white border border-secondary text-primary hover:bg-gray-50',
    outline: 'border border-accent text-accent hover:bg-accent hover:text-white',
    social: 'bg-gray-100 text-primary hover:bg-gray-200/80 transition-colors',
  };

  return (
    <motion.button
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.98 }}
      className={`w-full py-3 px-6 rounded-xl font-semibold transition-all duration-200 flex items-center justify-center gap-3 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </motion.button>
  );
};

export default Button;
