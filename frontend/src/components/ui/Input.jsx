import { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';

const Input = ({ label, labelRight, type = 'text', className = '', ...props }) => {
  const [showPassword, setShowPassword] = useState(false);
  const isPassword = type === 'password';

  return (
    <div className={`space-y-1.5 ${className}`}>
      {(label || labelRight) && (
        <div className="flex items-center justify-between">
          {label && (
            <label className="block text-sm font-bold text-primary">
              {label}
            </label>
          )}
          {labelRight && <div className="text-sm">{labelRight}</div>}
        </div>
      )}
      <div className="relative">
        <input
          type={isPassword ? (showPassword ? 'text' : 'password') : type}
          className="w-full px-4 py-3 bg-white border border-secondary rounded-xl focus:ring-2 focus:ring-accent/20 focus:border-accent outline-none transition-all placeholder:text-gray-400"
          {...props}
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-secondary hover:text-primary transition-colors"
          >
            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        )}
      </div>
    </div>
  );
};

export default Input;
