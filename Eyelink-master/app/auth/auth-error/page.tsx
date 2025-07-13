import { Button } from '@/components/ui/button'
import Link from 'next/link'

export default function AuthErrorPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="max-w-md w-full space-y-8 p-8 bg-card rounded-lg shadow-lg text-center">
        <h2 className="text-3xl font-bold">Authentication Error</h2>
        <div className="mt-4 text-muted-foreground">
          <p>There was a problem authenticating your account.</p>
          <p className="mt-2">Please try signing in again.</p>
        </div>
        <div className="mt-6">
          <Button asChild>
            <Link href="/auth/login">
              Return to Sign In
            </Link>
          </Button>
        </div>
      </div>
    </div>
  )
} 