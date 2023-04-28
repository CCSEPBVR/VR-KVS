/*****************************************************************************/
/**
 *  @file   FrameEventListener.cpp
 */
/*----------------------------------------------------------------------------
 *
 *  Copyright (c) Visualization Laboratory, Kyoto University.
 *  All rights reserved.
 *  See http://www.viz.media.kyoto-u.ac.jp/kvs/copyright/ for details.
 *
 *  $Id: PaintEventListener.cpp 1325 2012-10-04 10:34:52Z naohisa.sakamoto@gmail.com $
 */
/*****************************************************************************/
#include "FrameEventListener.h"
#include <kvs/IgnoreUnusedVariable>

namespace kvs
{

/*===========================================================================*/
/**
 *  @brief  Constructs a new FrameEventListener class.
 */
/*===========================================================================*/
FrameEventListener::FrameEventListener()
{
    kvs::EventListener::setEventType( kvs::EventBase::FrameEvent );
}

/*===========================================================================*/
/**
 *  @brief  Destructs the PaintEventListener class.
 */
/*===========================================================================*/
FrameEventListener::~FrameEventListener()
{
}

/*===========================================================================*/
/**
 *  @brief  Executes the event function.
 *  @param  event [in] pointer to the event
 */
/*===========================================================================*/
void FrameEventListener::onEvent( kvs::EventBase* event )
{
    kvs::IgnoreUnusedVariable( event );

    this->update();
}

} // end of namespace kvs
