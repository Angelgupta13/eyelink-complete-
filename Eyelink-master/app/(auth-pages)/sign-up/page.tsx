import { signUpAction } from "@/app/actions";
import { FormMessage, Message } from "@/components/form-message";
import { SubmitButton } from "@/components/submit-button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FaEye } from "react-icons/fa";
import Link from "next/link";


export default async function Signup(props: {
  searchParams: Promise<Message>;
}) {
  const searchParams = await props.searchParams;
  if ("message" in searchParams) {
    return (
      <div className="w-full flex-1 flex items-center justify-center p-4 py-">
        <div className="w-full max-w-md">
          <FormMessage message={searchParams} />
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md mx-auto space-y-10 p-4">
      <div className="flex flex-col items-center space-y-2">
        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
          <FaEye className="w-6 h-6 text-white" />
        </div>
        <h1 className="text-2xl font-bold tracking-tight">Create an account</h1>
        <p className="text-sm text-muted-foreground">Enter your details to get started</p>
      </div>

      <form className="space-y-6">
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              name="email"
              type="email"
              placeholder="you@example.com"
              required
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              name="password"
              placeholder="Your password"
              minLength={6}
              required
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Must be at least 6 characters long
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <SubmitButton
            className="w-full"
            formAction={signUpAction}
            pendingText="Signing up..."
          >
            Create account
          </SubmitButton>

          <FormMessage message={searchParams} />
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-muted" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-background px-2 text-muted-foreground">
              Already have an account?
            </span>
          </div>
        </div>

        <div className="text-center">
          <Link
            className="text-sm text-primary hover:text-primary/90 font-medium underline-offset-4 hover:underline"
            href="/sign-in"
          >
            Sign in
          </Link>
        </div>
      </form>

      
    </div>
  );
}
