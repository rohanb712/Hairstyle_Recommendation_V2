'use client'

import { useState, useEffect } from 'react'
import { User, Calendar, Zap, Palette } from 'lucide-react'
import { apiService } from '@/services/api'
import {
  UserProfileRequest,
  UserProfileResponse,
  EthnicityType,
  HairTextureType,
  GenderType,
  ProfileOptions,
} from '@/types/api'

interface UserProfileFormProps {
  onProfileCreated: (profile: UserProfileResponse) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export default function UserProfileForm({
  onProfileCreated,
  isLoading,
  setIsLoading,
}: UserProfileFormProps) {
  const [formData, setFormData] = useState<UserProfileRequest>({
    ethnicity: 'caucasian',
    age: 25,
    gender: 'female',
    hair_texture: 'straight',
  })

  const [profileOptions, setProfileOptions] = useState<ProfileOptions | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [step, setStep] = useState(1)

  // Fallback options to ensure the form renders even if the backend is down or the request fails
  const DEFAULT_PROFILE_OPTIONS: ProfileOptions = {
    ethnicities: [
      'african',
      'asian',
      'caucasian',
      'hispanic',
      'middle_eastern',
      'native_american',
      'pacific_islander',
      'south_asian',
      'mixed',
      'other',
    ],
    hair_textures: ['straight', 'wavy', 'curly', 'coily'],
    genders: ['male', 'female', 'non_binary'],
    age_range: { min: 13, max: 100 },
  }

  useEffect(() => {
    const loadProfileOptions = async () => {
      try {
        const options = await apiService.getProfileOptions()
        setProfileOptions(options)
      } catch (error) {
        console.error('Failed to load profile options:', error)
        // Use fallback values so the UI can still render
        setProfileOptions(DEFAULT_PROFILE_OPTIONS)
      }
    }

    loadProfileOptions()
  }, [])

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (!formData.ethnicity) {
      newErrors.ethnicity = 'Please select your ethnicity'
    }

    if (!formData.age || formData.age < 13 || formData.age > 100) {
      newErrors.age = 'Age must be between 13 and 100'
    }

    if (!formData.gender) {
      newErrors.gender = 'Please select your gender'
    }

    if (!formData.hair_texture) {
      newErrors.hair_texture = 'Please select your hair texture'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) {
      return
    }

    setIsLoading(true)
    try {
      const response = await apiService.createProfile(formData)
      onProfileCreated(response)
    } catch (error) {
      console.error('Failed to create profile:', error)
      setErrors({ submit: 'Failed to create profile. Please try again.' })
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (field: keyof UserProfileRequest, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: '' }))
    }
  }

  const nextStep = () => {
    if (step === 1 && (formData.ethnicity && formData.age >= 13 && formData.age <= 100)) {
      setStep(2)
    } else if (step === 2 && formData.gender) {
      setStep(3)
    }
  }

  const prevStep = () => {
    if (step > 1) {
      setStep(step - 1)
    }
  }

  const ethnicityLabels: Record<EthnicityType, string> = {
    african: 'African',
    asian: 'Asian',
    caucasian: 'Caucasian',
    hispanic: 'Hispanic/Latino',
    middle_eastern: 'Middle Eastern',
    native_american: 'Native American',
    pacific_islander: 'Pacific Islander',
    south_asian: 'South Asian',
    mixed: 'Mixed/Multiracial',
    other: 'Other',
  }

  const hairTextureLabels: Record<HairTextureType, string> = {
    straight: 'Straight',
    wavy: 'Wavy',
    curly: 'Curly',
    coily: 'Coily/Kinky',
  }

  const genderLabels: Record<GenderType, string> = {
    male: 'Male',
    female: 'Female',
    non_binary: 'Non-binary',
  }

  if (!profileOptions) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          Tell Us About Yourself
        </h2>
        <p className="text-gray-600">
          Help us provide personalized hairstyle recommendations
        </p>
      </div>

      {/* Progress Indicator */}
      <div className="flex justify-center items-center space-x-4 mb-8">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
          step >= 1 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-500'
        }`}>
          1
        </div>
        <div className={`w-16 h-1 ${step >= 2 ? 'bg-blue-300' : 'bg-gray-200'}`} />
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
          step >= 2 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-500'
        }`}>
          2
        </div>
        <div className={`w-16 h-1 ${step >= 3 ? 'bg-blue-300' : 'bg-gray-200'}`} />
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
          step >= 3 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-500'
        }`}>
          3
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Step 1: Demographics */}
        {step === 1 && (
          <div className="space-y-6">
            <div className="flex items-center space-x-3 mb-6">
              <User className="w-6 h-6 text-blue-600" />
              <h3 className="text-xl font-semibold text-gray-800">Demographics</h3>
            </div>

            {/* Ethnicity */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Ethnicity
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {profileOptions.ethnicities.map((ethnicity) => (
                  <button
                    key={ethnicity}
                    type="button"
                    onClick={() => handleInputChange('ethnicity', ethnicity)}
                    className={`p-3 text-left rounded-lg border-2 transition-all ${
                      formData.ethnicity === ethnicity
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-200 hover:border-gray-300 text-gray-700'
                    }`}
                  >
                    {ethnicityLabels[ethnicity]}
                  </button>
                ))}
              </div>
              {errors.ethnicity && (
                <p className="mt-1 text-sm text-red-600">{errors.ethnicity}</p>
              )}
            </div>

            {/* Age */}
            <div>
              <label htmlFor="age" className="block text-sm font-medium text-gray-700 mb-2">
                Age
              </label>
              <div className="flex items-center space-x-3">
                <Calendar className="w-5 h-5 text-gray-400" />
                <input
                  id="age"
                  type="number"
                  min="13"
                  max="100"
                  value={formData.age}
                  onChange={(e) => handleInputChange('age', parseInt(e.target.value) || 0)}
                  className={`flex-1 p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                    errors.age ? 'border-red-300' : 'border-gray-300'
                  }`}
                  placeholder="Enter your age"
                />
              </div>
              {errors.age && (
                <p className="mt-1 text-sm text-red-600">{errors.age}</p>
              )}
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={nextStep}
                disabled={!formData.ethnicity || formData.age < 13 || formData.age > 100}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Next Step
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Gender */}
        {step === 2 && (
          <div className="space-y-6">
            <div className="flex items-center space-x-3 mb-6">
              <User className="w-6 h-6 text-blue-600" />
              <h3 className="text-xl font-semibold text-gray-800">Gender Identity</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                How do you identify?
              </label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {profileOptions.genders.map((gender) => (
                  <button
                    key={gender}
                    type="button"
                    onClick={() => handleInputChange('gender', gender)}
                    className={`p-4 text-center rounded-lg border-2 transition-all ${
                      formData.gender === gender
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-200 hover:border-gray-300 text-gray-700'
                    }`}
                  >
                    <div className="font-medium">{genderLabels[gender]}</div>
                  </button>
                ))}
              </div>
              {errors.gender && (
                <p className="mt-1 text-sm text-red-600">{errors.gender}</p>
              )}
            </div>

            <div className="flex justify-between">
              <button
                type="button"
                onClick={prevStep}
                className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
              >
                Previous
              </button>
              <button
                type="button"
                onClick={nextStep}
                disabled={!formData.gender}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Next Step
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Hair Texture */}
        {step === 3 && (
          <div className="space-y-6">
            <div className="flex items-center space-x-3 mb-6">
              <Palette className="w-6 h-6 text-blue-600" />
              <h3 className="text-xl font-semibold text-gray-800">Hair Texture</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                What's your natural hair texture?
              </label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {profileOptions.hair_textures.map((texture) => (
                  <button
                    key={texture}
                    type="button"
                    onClick={() => handleInputChange('hair_texture', texture)}
                    className={`p-4 text-center rounded-lg border-2 transition-all ${
                      formData.hair_texture === texture
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-200 hover:border-gray-300 text-gray-700'
                    }`}
                  >
                    <Zap className="w-6 h-6 mx-auto mb-2" />
                    <div className="font-medium">{hairTextureLabels[texture]}</div>
                  </button>
                ))}
              </div>
              {errors.hair_texture && (
                <p className="mt-1 text-sm text-red-600">{errors.hair_texture}</p>
              )}
            </div>

            {errors.submit && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-600">{errors.submit}</p>
              </div>
            )}

            <div className="flex justify-between">
              <button
                type="button"
                onClick={prevStep}
                className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
              >
                Previous
              </button>
              <button
                type="submit"
                disabled={isLoading || !formData.hair_texture}
                className="px-8 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Creating Profile...</span>
                  </>
                ) : (
                  <span>Create Profile & Continue</span>
                )}
              </button>
            </div>
          </div>
        )}
      </form>
    </div>
  )
} 